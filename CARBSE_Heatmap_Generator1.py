# -*- coding: utf-8 -*-
"""
CARBSE Heatmap Generator (Type 1)
- Modern UI + all classic controls restored
- Works with single worker (recommended) and session-safe uploads
"""

import os
import io
import uuid
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, session, flash
)
from flask_caching import Cache
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
# If your templates folder is capitalized on Windows, keep the default lower-case
# on Linux. Make sure your repo path is: templates/heatmap_app_00.html
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")

# Simple in-memory cache is fine for single worker
cache = Cache(config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 60 * 60})
cache.init_app(app)

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
REQUIRED_COLS = ["Timestamp", "DBT", "RH"]

DEFAULTS = {
    "DBT": {"zmin": 0.0, "zmax": 50.0},
    "RH": {"zmin": 0.0, "zmax": 100.0},
}

# Exact palette requested (White → Navyblue → Blue → Cyan → Yellow → Red → Darkred → Black)
COLOR_MAP = {
    "White":   "#FFFFFF",
    "Navyblue": "#001f3f",
    "Blue":    "#0074D9",
    "Cyan":    "#00FFFF",
    "Yellow":  "#FFDC00",
    "Red":     "#FF4136",
    "Darkred": "#8B0000",
    "Black":   "#000000",
}
FULL_SCALE_NAMES = ["White", "Navyblue", "Blue", "Cyan", "Yellow", "Red", "Darkred", "Black"]

def make_colorscale(names):
    # even stops across 0..1
    if len(names) == 1:
        return [[0, COLOR_MAP[names[0]]], [1, COLOR_MAP[names[0]]]]
    stops = []
    n = len(names) - 1
    for i, nm in enumerate(names):
        stops.append([i / n, COLOR_MAP[nm]])
    return stops

DATE_TICK_FORMATS = {
    "mon2yr": "%b-%y",          # Jan-17
    "dd-mon": "%d-%b",          # 01-Jan
    "dd-mm-yy": "%d-%m-%y",     # 01-01-17
    "dd-fullmon": "%d-%B",      # 01-January
    "wk-abbr": "%a",            # Mon
    "wk-full": "%A",            # Monday
    "mon-abbr": "%b",           # Jan
    "mon-01": "%m",             # 01
    "yy": "%y",                 # 17
    "yyyy": "%Y",               # 2017
}

def _session_key():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return f"upload:{session['sid']}"

def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Parse timestamp safely
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    if ts.isna().all():
        # try without dayfirst
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if ts.isna().any():
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]
    df["Timestamp"] = ts
    df["Date"] = df["Timestamp"].dt.normalize()
    df["Hour"] = df["Timestamp"].dt.hour
    return df

def _read_upload(file_storage) -> pd.DataFrame:
    filename = file_storage.filename
    data = file_storage.read()
    if not data:
        raise ValueError("Empty file.")
    buf = io.BytesIO(data)
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)
    return _coerce_df(df)

def _pivot(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    # daily (columns) × hour (rows)
    pt = df.pivot_table(index="Hour", columns="Date", values=parameter, aggfunc="mean")
    # ensure clean axes
    pt = pt.sort_index(axis=0).sort_index(axis=1)
    # keep full 0..23 even if missing (fills NaN)
    pt = pt.reindex(range(0, 24))
    return pt

def _build_fig(pt: pd.DataFrame, parameter: str, zmin: float, zmax: float,
               date_fmt_key: str, tickfont_size: int, height: int):
    colorscale = make_colorscale(FULL_SCALE_NAMES)
    fig = go.Figure(
        data=go.Heatmap(
            x=pt.columns, y=pt.index, z=pt.values,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=dict(title=parameter),
            hovertemplate=f"Date: %{{x}}<br>Hour: %{{y}}<br>{parameter}: %{{z:.2f}}<extra></extra>",
        )
    )
    # D3 tick format
    d3fmt = DATE_TICK_FORMATS.get(date_fmt_key, "%b-%y")

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=60, b=80),
        title=dict(text=f"Single-City Heatmap • {parameter}", x=0.02, xanchor="left"),
        xaxis=dict(
            type="date",
            tickformat=d3fmt,
            tickfont=dict(size=tickfont_size),
            title="Date"
        ),
        yaxis=dict(
            title="Hour (0–23)",
            tickmode="linear",
            tick0=0,
            dtick=1,
            tickfont=dict(size=12),
        ),
        # keep chart steady on re-render
        uirevision="keep",
    )
    return fig

def _save_image(fig: go.Figure, width: int, height: int) -> str:
    os.makedirs("/tmp/plots", exist_ok=True)
    pid = uuid.uuid4().hex
    path = f"/tmp/plots/{pid}.jpg"
    fig.write_image(path, format="jpg", width=width, height=height, scale=1)
    return pid, path

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    # Retrieve prior state (if any)
    key = _session_key()
    state = cache.get(key) or {}

    return render_template(
        "heatmap_app_00.html",
        uploaded_name=state.get("filename"),
        preview_rows=state.get("preview_rows", []),
        preview_cols=state.get("preview_cols", 0),
        plot_html=state.get("plot_html"),
        plot_id=state.get("plot_id"),
        defaults=DEFAULTS,
        # UI default selections
        selected_param=state.get("parameter", "DBT"),
        scalemin=state.get("zmin"),
        scalemax=state.get("zmax"),
        date_fmt_key=state.get("date_fmt_key", "mon2yr"),
        img_width=state.get("img_width", 1200),
        img_height=state.get("img_height", 600),
        tick_size=state.get("tick_size", 10),
    )

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a CSV or Excel file.", "warning")
        return redirect(url_for("home"))

    try:
        df = _read_upload(f)
    except Exception as e:
        flash(f"Upload failed: {e}", "danger")
        return redirect(url_for("home"))

    # Cache the cleaned dataframe for this session
    key = _session_key()
    state = {
        "filename": f.filename,
        "df_json": df.to_json(date_format="iso", orient="split"),
        "preview_rows": df[REQUIRED_COLS].head(30).values.tolist(),
        "preview_cols": len(df.columns),
    }
    cache.set(key, state)

    flash("File uploaded successfully.", "success")
    return redirect(url_for("home"))

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    key = _session_key()
    state = cache.get(key) or {}
    if "df_json" not in state:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("home"))

    df = pd.read_json(io.StringIO(state["df_json"]), orient="split")

    parameter = request.form.get("parameter", "DBT")
    zmin = request.form.get("scalemin", "").strip()
    zmax = request.form.get("scalemax", "").strip()
    img_width = int(request.form.get("img_width", 1200))
    img_height = int(request.form.get("img_height", 600))
    tick_size = int(request.form.get("tick_size", 10))
    date_fmt_key = request.form.get("date_fmt", "mon2yr")

    # Default scale per variable if empty
    if zmin == "":
        zmin = DEFAULTS.get(parameter, {}).get("zmin", 0.0)
    else:
        zmin = float(zmin)

    if zmax == "":
        zmax = DEFAULTS.get(parameter, {}).get("zmax", 100.0)
    else:
        zmax = float(zmax)

    # Pivot data
    pt = _pivot(df, parameter)

    # Build figure (on-page figure height a bit smaller for nice fit)
    fig = _build_fig(pt, parameter, zmin, zmax, date_fmt_key, tick_size, height=480)

    # HTML for inline plot (responsive)
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        default_height="100%"
    )

    # Save a JPG for download in requested image size
    pid, path = _save_image(_build_fig(pt, parameter, zmin, zmax, date_fmt_key, tick_size, height=img_height),
                            width=img_width, height=img_height)

    # Update state & return
    state.update({
        "plot_html": plot_html,
        "plot_id": pid,
        "parameter": parameter,
        "zmin": zmin, "zmax": zmax,
        "date_fmt_key": date_fmt_key,
        "img_width": img_width,
        "img_height": img_height,
        "tick_size": tick_size,
    })
    cache.set(key, state)
    return redirect(url_for("home"))

@app.route("/download/jpg/<plot_id>", methods=["GET"])
def download_jpg(plot_id):
    path = f"/tmp/plots/{plot_id}.jpg"
    if not os.path.exists(path):
        flash("Image not found. Please re-generate the heatmap.", "warning")
        return redirect(url_for("home"))
    return send_file(path, mimetype="image/jpeg", as_attachment=True, download_name="heatmap.jpg")

# -----------------------------------------------------------------------------
# Entrypoint (for local runs)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
