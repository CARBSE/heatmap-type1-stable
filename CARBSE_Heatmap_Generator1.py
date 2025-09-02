import os
import uuid
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file
)
import plotly.graph_objects as go
import plotly.io as pio

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "carbse-heatmap-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True

PLOT_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# In-memory cache per session (OK with 1 gunicorn worker)
CACHE: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def get_sid() -> str:
    """Get (or set) a simple session id."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse a 'Timestamp' column robustly.
    """
    # Try fast path
    dt = pd.to_datetime(series, errors="coerce")
    # If too many NaT, try dayfirst=True
    nat_ratio = dt.isna().mean()
    if nat_ratio > 0.3:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt

COLOR_MAP = {
    "White":    "#FFFFFF",
    "Navyblue": "#001f3f",
    "Blue":     "#0074D9",
    "Cyan":     "#00FFFF",
    "Yellow":   "#FFFF00",
    "Red":      "#FF4136",
    "Darkred":  "#990000",
    "Black":    "#000000",
}

MISSING_COLORS = {
    "Grey":  "#BDBDBD",
    "White": "#FFFFFF",
    "Black": "#000000",
}

DATE_FMT = {
    "mon2y": "%b-%y",           # Jan-17
    "ddmon": "%d-%b",           # 01-Jan
    "ddmonfull": "%d-%B",       # 01-January
    "mon3": "%b",               # Jan
    "dow3": "%a",               # Mon
    "dowfull": "%A",            # Monday
    "mm": "%m",                 # 01
    "yyyy": "%Y",               # 2017
    "ddmmyy": "%d-%m-%y",       # 01-01-17
}

def build_colorscale(selected: List[str]) -> List[List[Any]]:
    """Plotly colorscale from ordered color names."""
    if not selected:
        selected = ["White", "Blue", "Cyan", "Yellow", "Red", "Black"]
    cs = [COLOR_MAP.get(c, "#000000") for c in selected]
    if len(cs) == 1:
        cs = [cs[0], cs[0]]
    return [[i / (len(cs) - 1), c] for i, c in enumerate(cs)]

def human_tick_text(dates: List[pd.Timestamp], fmt_key: str) -> (List[str], List[pd.Timestamp]):
    fmt = DATE_FMT.get(fmt_key, "%b-%y")
    ticktext = [d.strftime(fmt) for d in dates]
    return ticktext, dates

def add_band_line(fig: go.Figure, hour: float, style: str, color: str):
    if hour is None:
        return
    dash = "dot" if style == "dotted" else "solid"
    fig.add_hline(y=hour, line_dash=dash, line_color=color, opacity=0.8)

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    sid = get_sid()
    data = CACHE.get(sid, {})
    df = data.get("df")

    data_html = None
    detected = []
    if df is not None:
        preview = df.head(100).copy()
        data_html = preview.to_html(
            classes="table table-striped table-sm table-hover mb-0",
            index=False, border=0
        )
        detected = list(df.columns)

    return render_template(
        "heatmap_app_00.html",
        has_data=df is not None,
        detected_cols=detected,
        data_html=data_html,
        plot_html=None,
        last_plot_id=None,
    )

@app.route("/upload", methods=["POST"])
def upload():
    sid = get_sid()
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("home"))

    try:
        if file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception:
        # try with cp1252 fallback
        file.stream.seek(0)
        df = pd.read_csv(file, encoding="cp1252")

    # normalize expected columns
    # Expected headers: Timestamp, DBT, RH   (case-insensitive accepted)
    lower = {c.lower(): c for c in df.columns}
    if "timestamp" in lower and lower["timestamp"] != "Timestamp":
        df.rename(columns={lower["timestamp"]: "Timestamp"}, inplace=True)
    if "dbt" in lower and lower["dbt"] != "DBT":
        df.rename(columns={lower["dbt"]: "DBT"}, inplace=True)
    if "rh" in lower and lower["rh"] != "RH":
        df.rename(columns={lower["rh"]: "RH"}, inplace=True)

    if "Timestamp" not in df.columns:
        # try "Date" or "Datetime"
        for cand in ["Date", "Datetime", "Time", "time"]:
            if cand in df.columns:
                df.rename(columns={cand: "Timestamp"}, inplace=True)
                break

    # parse timestamp
    df["Timestamp"] = parse_timestamp(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    CACHE[sid] = {
        "df": df,
        "fname": file.filename,
        "last_plot_id": None,
    }
    return redirect(url_for("home"))

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    sid = get_sid()
    data = CACHE.get(sid, {})
    df = data.get("df")
    if df is None:
        return redirect(url_for("home"))

    # ----- form values -----
    param = request.form.get("variable", "DBT")
    legend_name = request.form.get("legend_name") or param

    # scaling
    vmin = request.form.get("scale_min", "").strip()
    vmax = request.form.get("scale_max", "").strip()
    try:
        zmin = float(vmin) if vmin != "" else 0.0
    except ValueError:
        zmin = 0.0
    if vmax == "":
        zmax = 50.0 if param.upper() == "DBT" else 100.0
    else:
        try:
            zmax = float(vmax)
        except ValueError:
            zmax = 50.0 if param.upper() == "DBT" else 100.0

    # sizes (used only for JPG)
    img_w = int(request.form.get("image_width") or 1200)
    img_h = int(request.form.get("image_height") or 600)

    # bands
    def _parse_time(s):
        s = (s or "").strip()
        if not s:
            return None
        try:
            t = datetime.strptime(s, "%H:%M")
            return t.hour + t.minute / 60.0
        except Exception:
            return None

    upper_time = _parse_time(request.form.get("upper_band_time"))
    upper_style = request.form.get("upper_band_style", "dotted")
    upper_color = request.form.get("upper_band_color", "black")

    lower_time = _parse_time(request.form.get("lower_band_time"))
    lower_style = request.form.get("lower_band_style", "dotted")
    lower_color = request.form.get("lower_band_color", "black")

    # colors
    missing_color_name = request.form.get("missing_color", "Grey")
    missing_color = MISSING_COLORS.get(missing_color_name, "#BDBDBD")
    selected_colors = request.form.getlist("colors")
    colorscale = build_colorscale(selected_colors)

    # text
    title = request.form.get("title") or f"Heatmap ({param})"
    x_name = request.form.get("x_name") or "Date"
    y_name = request.form.get("y_name") or "Hour"

    # date axis
    date_fmt_key = request.form.get("date_fmt", "mon2y")
    x_repr = request.form.get("x_repr", "months")   # months | days

    # ------------- pivot -------------
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DateOnly"] = df["Timestamp"].dt.floor("D")

    # Aggregate to mean per Hour x Date
    pv = pd.pivot_table(
        df, index="Hour", columns="DateOnly", values=param, aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    hours = pv.index.to_list()
    dates = list(pv.columns)  # Timestamp (midnight)
    z = pv.values

    # ------------- plotting -------------
    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        x=dates,
        y=hours,
        z=z,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=legend_name),
        hoverongaps=False,
        hovertemplate=f"Date: %{{x|%b %d, %Y}}<br>Hour: %{{y}}:00<br>{legend_name}: %{{z:.2f}}<extra></extra>",
        showscale=True,
    ))

    # Overlay for NaN cells with a single-color heatmap (does not change scale)
    nan_mask = ~np.isfinite(z)
    if nan_mask.any():
        z_nan = np.where(nan_mask, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            x=dates,
            y=hours,
            z=z_nan,
            colorscale=[[0, missing_color], [1, missing_color]],
            showscale=False,
            hoverinfo="skip",
            opacity=1.0
        ))

    # bands
    add_band_line(fig, upper_time, upper_style, upper_color)
    add_band_line(fig, lower_time, lower_style, lower_color)

    # layout (web: responsive width; jpg: fixed)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title=x_name, ticks="outside"),
        yaxis=dict(title=y_name, range=[-0.5, 23.5]),
        width=None,           # responsive
        height=img_h,
        autosize=True,
        margin=dict(l=60, r=60, t=50, b=60),
        template="plotly_white",
    )

    # X-axis ticks
    if x_repr == "months":
        fig.update_xaxes(dtick="M1")
    else:
        fig.update_xaxes(dtick="D1")

    # custom tick text formatting
    if dates:
        ticktext, tickvals = human_tick_text(dates, date_fmt_key)
        fig.update_xaxes(ticktext=ticktext, tickvals=tickvals)

    # HTML (responsive)
    graph_html = pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False}
    )

    # JPG
    plot_id = str(uuid.uuid4())
    jpg_path = os.path.join(PLOT_DIR, f"{plot_id}.jpg")
    fig_img = go.Figure(fig)
    fig_img.update_layout(width=img_w, height=img_h)
    # scale=2 for crisper image
    fig_img.write_image(jpg_path, scale=2)

    # update cache & render
    data["last_plot_id"] = plot_id
    CACHE[sid] = data

    # preview table again (Data tab)
    preview = CACHE[sid]["df"].head(100)
    data_html = preview.to_html(
        classes="table table-striped table-sm table-hover mb-0",
        index=False, border=0
    )
    detected = list(CACHE[sid]["df"].columns)

    return render_template(
        "heatmap_app_00.html",
        has_data=True,
        detected_cols=detected,
        data_html=data_html,
        plot_html=graph_html,
        last_plot_id=plot_id,
    )

@app.route("/download/jpg/<plot_id>")
def download_jpg(plot_id):
    file_path = os.path.join(PLOT_DIR, f"{plot_id}.jpg")
    if not os.path.exists(file_path):
        return redirect(url_for("home"))
    return send_file(file_path, as_attachment=True, download_name="heatmap.jpg")

# -------------------------------------------------------
# Local run
# -------------------------------------------------------
if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
