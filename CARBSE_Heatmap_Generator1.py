import os
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Flask
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="Templates")
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "change-me"),  # set in Render for security
    SESSION_COOKIE_SAMESITE="Lax",
)

# Directories (ephemeral on Render but persists across requests of the same deploy)
BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
PERSIST_DIR = "/tmp/carbse-heatmap"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
COLOR_MAP = {
    "White": "#FFFFFF",
    "Navyblue": "#001f3f",
    "Blue": "#1f77b4",
    "Cyan": "#17becf",
    "Yellow": "#FFDC00",
    "Red": "#FF4136",
    "Darkred": "#8B0000",
    "Black": "#111111",
}

LINE_STYLES = {"solid": "solid", "dashed": "dash", "dotted": "dot"}

DATE_TICKFORMAT = {
    "mon_2digit_year": "%b-%y",
    "dd_mon": "%d-%b",
    "dd_mm_yy": "%d-%m-%y",
    "dd_month": "%d-%B",
    "weekday_abbr": "%a",
    "weekday_full": "%A",
    "month_abbr": "%b",
    "month_dec": "%m",
    "yy_2digit": "%y",
    "yyyy_4digit": "%Y",
}

def sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def df_path() -> str:
    return os.path.join(PERSIST_DIR, f"{sid()}.pkl")

def save_df(df: pd.DataFrame):
    # store to /tmp per-session
    df.to_pickle(df_path())

def load_df() -> pd.DataFrame | None:
    try:
        path = df_path()
        if os.path.exists(path):
            return pd.read_pickle(path)
    except Exception:
        return None
    return None

def parse_timestamp(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().mean() > 0.3:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt

def colors_to_scale(names: list[str]):
    if not names:
        names = ["Blue", "Cyan", "Yellow", "Red"]
    hexes = [COLOR_MAP.get(n, n) for n in names]
    steps = np.linspace(0, 1, len(hexes)).tolist()
    return [[float(s), c] for s, c in zip(steps, hexes)]

def build_heatmap(
    df: pd.DataFrame,
    parameter: str,
    zmin: float | None,
    zmax: float | None,
    colors: list[str],
    nodata_color: str,
    upper_time: str | None,
    upper_style: str,
    upper_color: str,
    lower_time: str | None,
    lower_style: str,
    lower_color: str,
    legend_step: float | None,
    tickformat_key: str,
    x_representation: str,
    title: str,
    xname: str,
    yname: str,
    lname: str,
    width: int,
    height: int,
    base_font: int,
):
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'Timestamp' column.")
    if parameter not in df.columns:
        raise ValueError(f"CSV must contain '{parameter}' column.")

    df = df.copy()
    df["Timestamp"] = parse_timestamp(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"]  = df["Timestamp"].dt.floor("D")

    pivot = df.pivot_table(index="Hour", columns="Day", values=parameter, aggfunc="mean")
    pivot = pivot.sort_index(axis=1)

    # Coloraxis (lets us set nancolor)
    colorscale = colors_to_scale(colors)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            coloraxis="coloraxis",
            connectgaps=False,
            hovertemplate=f"Date: %{{x}}<br>Hour: %{{y}}<br>{parameter}: %{{z:.2f}}<extra></extra>",
            showscale=True,
        )
    )

    # Layout & axes
    tickformat = DATE_TICKFORMAT.get(tickformat_key, "%b-%y")
    dtick = None
    if x_representation == "months":   dtick = "M1"
    elif x_representation == "quarters": dtick = "M3"
    elif x_representation == "years":  dtick = "M12"
    elif x_representation == "weeks":  dtick = 7*24*3600*1000
    elif x_representation == "days":   dtick = 24*3600*1000

    cbar = {}
    if legend_step and legend_step > 0:
        cbar = dict(tickmode="linear", dtick=legend_step)

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            nancolor=nodata_color,
            colorbar=dict(title=lname or parameter, **cbar),
        ),
        title=title or f"Single-City Heatmap • {parameter}",
        xaxis=dict(title=xname or "Date", type="date", tickformat=tickformat, dtick=dtick, showgrid=False),
        yaxis=dict(title=yname or "Hour", tickmode="linear", tick0=0, dtick=1),
        margin=dict(l=60, r=30, t=60, b=60),
        width=width,
        height=height,
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#FFFFFF",
        font=dict(size=base_font, family="Inter, system-ui, Segoe UI, Roboto, Helvetica, Arial"),
    )

    # Band lines
    def hh_to_int(hhmm: str | None):
        if not hhmm: return None
        try:
            return int(hhmm.split(":")[0])
        except Exception:
            return None

    x0 = pivot.columns.min() if not pivot.empty else None
    x1 = pivot.columns.max() if not pivot.empty else None
    if x0 is not None and x1 is not None:
        up = hh_to_int(upper_time)
        lo = hh_to_int(lower_time)
        if up is not None:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=up, y1=up,
                          line=dict(color=upper_color or "black", width=1.5, dash=LINE_STYLES.get(upper_style,"dot")),
                          xref="x", yref="y")
        if lo is not None:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=lo, y1=lo,
                          line=dict(color=lower_color or "black", width=1.5, dash=LINE_STYLES.get(lower_style,"dot")),
                          xref="x", yref="y")
    return fig

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    df = load_df()
    preview = df.head(30).to_dict(orient="records") if df is not None else None
    cols = list(df.columns) if df is not None else []
    return render_template("heatmap_app_00.html",
                           has_data=df is not None,
                           preview=preview, columns=cols,
                           plot_html=None, plot_id=None)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a CSV or Excel file.", "warning")
        return redirect(url_for("home"))

    name = f.filename.lower()
    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)
    except Exception as e:
        flash(f"Failed to read file: {e}", "danger")
        return redirect(url_for("home"))

    df.columns = [c.strip() for c in df.columns]
    save_df(df)
    flash("File uploaded successfully.", "success")
    return redirect(url_for("home"))

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    df = load_df()
    if df is None:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("home"))

    parameter = request.form.get("parameter", "DBT")
    smin = request.form.get("scale_min", "").strip()
    smax = request.form.get("scale_max", "").strip()

    zmin = float(smin) if smin != "" else None
    if smax == "":
        zmax = 100.0 if parameter.upper() == "RH" else 50.0
    else:
        zmax = float(smax)

    colors = request.form.getlist("colors")
    nodata_color = request.form.get("missing_color", "grey")

    upper_time = request.form.get("upper_time", "")
    upper_style= request.form.get("upper_style", "dotted")
    upper_color= request.form.get("upper_color", "black")
    lower_time = request.form.get("lower_time", "")
    lower_style= request.form.get("lower_style", "dotted")
    lower_color= request.form.get("lower_color", "black")

    legend_step = request.form.get("legend_step", "").strip()
    legend_step = float(legend_step) if legend_step else None

    tickformat_key = request.form.get("date_format", "mon_2digit_year")
    x_repr = request.form.get("x_representation", "months")

    title = request.form.get("title", "")
    xname = request.form.get("xname", "Date")
    yname = request.form.get("yname", "Hour")
    lname = request.form.get("lname", parameter)

    width = int(float(request.form.get("img_w", "1200") or 1200))
    height= int(float(request.form.get("img_h", "600")  or 600))
    base_font = int(float(request.form.get("point_size", "15") or 15))

    try:
        fig = build_heatmap(df, parameter, zmin, zmax, colors, nodata_color,
                            upper_time, upper_style, upper_color,
                            lower_time, lower_style, lower_color,
                            legend_step, tickformat_key, x_repr,
                            title, xname, yname, lname, width, height, base_font)
    except Exception as e:
        flash(f"Failed to build heatmap: {e}", "danger")
        return redirect(url_for("home"))

    # Save image for download
    plot_id = str(uuid.uuid4())
    jpg_path = os.path.join(PLOTS_DIR, f"{plot_id}.jpg")
    try:
        fig.write_image(jpg_path, format="jpg", width=width, height=height, scale=2, engine="kaleido")
    except Exception as e:
        flash(f"Interactive plot ready; JPG export error: {e}", "warning")

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    preview = df.head(30).to_dict(orient="records")
    cols = list(df.columns)
    return render_template("heatmap_app_00.html",
                           has_data=True,
                           preview=preview, columns=cols,
                           plot_html=plot_html, plot_id=plot_id)

@app.route("/download/jpg/<plot_id>")
def download_jpg(plot_id):
    path = os.path.join(PLOTS_DIR, f"{plot_id}.jpg")
    if not os.path.exists(path):
        flash("Image not found — please generate again.", "warning")
        return redirect(url_for("home"))
    return send_file(path, as_attachment=True, download_name="heatmap.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
