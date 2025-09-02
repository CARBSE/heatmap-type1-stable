import os
import uuid
from datetime import datetime
from typing import List

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file
)
from flask_caching import Cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Flask & Cache
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="Templates")
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "change-me"),
    CACHE_TYPE="SimpleCache",
    CACHE_DEFAULT_TIMEOUT=60 * 60,  # 1 hour
)
cache = Cache(app)

# folders
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    "mon_2digit_year": "%b-%y",             # Abbreviated month - 2-digit year (Jan-17)
    "dd_mon": "%d-%b",                       # Decimal date - Abbreviated month (01-Jan)
    "dd_mm_yy": "%d-%m-%y",                  # Decimal date - 2 digit year (01-01-17)
    "dd_month": "%d-%B",                     # Decimal date - Full month (01-January)
    "weekday_abbr": "%a",                    # Abbreviated weekday (Mon)
    "weekday_full": "%A",                    # Full weekday (Monday)
    "month_abbr": "%b",                      # Abbreviated month (Jan)
    "month_dec": "%m",                       # Decimal month (01)
    "yy_2digit": "%y",                       # 2-digit year (17)
    "yyyy_4digit": "%Y",                     # 4-digit year (2017)
}

def get_sid() -> str:
    """Simple per-session key to keep the uploaded DF around."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def store_df(df: pd.DataFrame):
    cache.set(f"df::{get_sid()}", df)

def load_df() -> pd.DataFrame | None:
    return cache.get(f"df::{get_sid()}")

def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parsing (day-first tolerant).
    """
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().mean() > 0.3:
        # try day-first
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt

def colors_to_plotly_scale(colors: List[str]):
    """
    Convert a list of hex colors into a Plotly continuous colorscale
    spread evenly from 0..1
    """
    if not colors:
        colors = [COLOR_MAP[c] for c in ["Blue", "Cyan", "Yellow", "Red"]]
    hexes = [COLOR_MAP.get(c, c) for c in colors]
    steps = np.linspace(0, 1, len(hexes))
    return [[float(s), h] for s, h in zip(steps, hexes)]

def build_heatmap(
    df: pd.DataFrame,
    parameter: str,
    zmin: float | None,
    zmax: float | None,
    colors: List[str],
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
    # Ensure Timestamp
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'Timestamp' column.")

    ts = parse_timestamp(df["Timestamp"])
    df = df.copy()
    df["Timestamp"] = ts
    df = df.dropna(subset=["Timestamp"])

    # Focus column
    if parameter not in df.columns:
        raise ValueError(f"CSV must contain '{parameter}' column.")

    # Hour as 0..23
    df["Hour"] = df["Timestamp"].dt.hour

    # Bucket X by day (you can change to month/week if desired)
    df["Day"] = df["Timestamp"].dt.floor("D")

    # Pivot: average per day-hour
    pivot = df.pivot_table(index="Hour", columns="Day", values=parameter, aggfunc="mean")

    # Sort columns by date
    pivot = pivot.sort_index(axis=1)

    # Build figure
    colorscale = colors_to_plotly_scale(colors)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=lname or parameter, tickmode="auto"),
            hovertemplate=f"Date: %{{x}}<br>Hour: %{{y}}<br>{parameter}: %{{z:.2f}}<extra></extra>",
            showscale=True,
            connectgaps=False,
            hoverongaps=True,
        )
    )

    # Missing colour (plotly handles NaNs via colorscale 'nan' color in layout)
    fig.update_traces(nicolor=nodata_color)

    # Axis tick format
    tickformat = DATE_TICKFORMAT.get(tickformat_key, "%b-%y")

    # X representation (monthly ticks, weekly, etc.)
    # plotly dtick docs: "M1" for monthly, "M3" for quarterly; milliseconds for fixed spacing
    dtick = None
    if x_representation == "months":
        dtick = "M1"
    elif x_representation == "quarters":
        dtick = "M3"
    elif x_representation == "years":
        dtick = "M12"
    elif x_representation == "weeks":
        # 1 week in ms
        dtick = 7 * 24 * 3600 * 1000
    elif x_representation == "days":
        dtick = 24 * 3600 * 1000

    # Legend tick step
    cbar = {}
    if legend_step and legend_step > 0:
        cbar = dict(
            tickmode="linear",
            tick0=zmin if zmin is not None else None,
            dtick=legend_step
        )

    fig.data[0].update(colorbar=cbar or fig.data[0].colorbar)

    # Layout
    fig.update_layout(
        title=title or f"Single-City Heatmap • {parameter}",
        xaxis=dict(
            title=xname or "Date",
            type="date",
            tickformat=tickformat,
            dtick=dtick,
            showgrid=False,
            ticks="outside",
            tickfont=dict(size=base_font-1),
        ),
        yaxis=dict(
            title=yname or "Hour",
            tickmode="linear",
            ticks="outside",
            tick0=0,
            dtick=1,
            tickfont=dict(size=base_font-1),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        width=width,
        height=height,
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#FFFFFF",
        font=dict(size=base_font, family="Inter, system-ui, Segoe UI, Roboto, Helvetica, Arial"),
    )

    # Band limit lines (if times provided)
    def parse_hhmm(hhmm: str | None):
        if not hhmm:
            return None
        try:
            h, m = hhmm.split(":")
            return int(h)
        except Exception:
            return None

    x0 = pivot.columns.min() if not pivot.empty else None
    x1 = pivot.columns.max() if not pivot.empty else None

    if x0 is not None and x1 is not None:
        up_h = parse_hhmm(upper_time)
        low_h = parse_hhmm(lower_time)

        if up_h is not None:
            fig.add_shape(
                type="line",
                x0=x0, x1=x1,
                y0=up_h, y1=up_h,
                line=dict(color=upper_color or "black", width=1.5, dash=LINE_STYLES.get(upper_style, "dot")),
                xref="x", yref="y"
            )
        if low_h is not None:
            fig.add_shape(
                type="line",
                x0=x0, x1=x1,
                y0=low_h, y1=low_h,
                line=dict(color=lower_color or "black", width=1.5, dash=LINE_STYLES.get(lower_style, "dot")),
                xref="x", yref="y"
            )

    return fig

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    df = load_df()
    preview = None
    cols = []
    if df is not None:
        preview = df.head(30).to_dict(orient="records")
        cols = list(df.columns)

    return render_template(
        "heatmap_app_00.html",
        has_data=df is not None,
        preview=preview,
        columns=cols,
        plot_html=None,
        plot_id=None,
    )

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or f.filename.strip() == "":
        flash("Please choose a CSV or Excel file.", "warning")
        return redirect(url_for("home"))

    filename = f.filename.lower()
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(f)
        else:
            # Allow a wide variety of CSVs
            df = pd.read_csv(f)
    except Exception as e:
        flash(f"Failed to read file: {e}", "danger")
        return redirect(url_for("home"))

    # normalize column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    store_df(df)
    flash("File uploaded successfully.", "success")
    return redirect(url_for("home"))

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    df = load_df()
    if df is None:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("home"))

    # Form values
    parameter = request.form.get("parameter", "DBT")
    scale_min = request.form.get("scale_min", "").strip()
    scale_max = request.form.get("scale_max", "").strip()

    # defaults for DBT/RH
    if scale_min == "":
        scale_min = None
    else:
        scale_min = float(scale_min)

    if scale_max == "":
        if parameter.upper() == "RH":
            scale_max = 100.0
        else:
            scale_max = 50.0
    else:
        scale_max = float(scale_max)

    # Colors & options
    selected_colors = request.form.getlist("colors")  # list of names
    nodata_color = request.form.get("missing_color", "grey")

    # Band lines
    upper_time = request.form.get("upper_time", "")
    upper_style = request.form.get("upper_style", "dotted")
    upper_color = request.form.get("upper_color", "black")

    lower_time = request.form.get("lower_time", "")
    lower_style = request.form.get("lower_style", "dotted")
    lower_color = request.form.get("lower_color", "black")

    # Legend tick step
    legend_step = request.form.get("legend_step", "").strip()
    legend_step = float(legend_step) if legend_step else None

    # Date format & representation
    tickformat_key = request.form.get("date_format", "mon_2digit_year")
    x_representation = request.form.get("x_representation", "months")

    # Texts
    title = request.form.get("title", "")
    xname = request.form.get("xname", "Date")
    yname = request.form.get("yname", "Hour")
    lname = request.form.get("lname", parameter)

    # Image / fonts
    width = int(float(request.form.get("img_w", "1200") or 1200))
    height = int(float(request.form.get("img_h", "600") or 600))
    base_font = int(float(request.form.get("point_size", "15") or 15))

    # Build figure
    try:
        fig = build_heatmap(
            df=df,
            parameter=parameter,
            zmin=scale_min,
            zmax=scale_max,
            colors=selected_colors,
            nodata_color=nodata_color,
            upper_time=upper_time,
            upper_style=upper_style,
            upper_color=upper_color,
            lower_time=lower_time,
            lower_style=lower_style,
            lower_color=lower_color,
            legend_step=legend_step,
            tickformat_key=tickformat_key,
            x_representation=x_representation,
            title=title,
            xname=xname,
            yname=yname,
            lname=lname,
            width=width,
            height=height,
            base_font=base_font,
        )
    except Exception as e:
        flash(f"Failed to build heatmap: {e}", "danger")
        return redirect(url_for("home"))

    # Save image to disk (for download)
    plot_id = str(uuid.uuid4())
    out_path = os.path.join(PLOTS_DIR, f"{plot_id}.jpg")
    try:
        fig.write_image(out_path, format="jpg", width=width, height=height, scale=2, engine="kaleido")
    except Exception as e:
        # Don’t block showing interactive plot if image save failed
        flash(f"Saved interactive plot; JPG export error: {e}", "warning")

    # Interactive embed
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    # Render with preview and data tabs
    preview = df.head(30).to_dict(orient="records")
    cols = list(df.columns)
    return render_template(
        "heatmap_app_00.html",
        has_data=True,
        preview=preview,
        columns=cols,
        plot_html=plot_html,
        plot_id=plot_id,
    )

@app.route("/download/jpg/<plot_id>", methods=["GET"])
def download_jpg(plot_id):
    path = os.path.join(PLOTS_DIR, f"{plot_id}.jpg")
    if not os.path.exists(path):
        flash("Image not found — please generate again.", "warning")
        return redirect(url_for("home"))
    return send_file(path, as_attachment=True, download_name="heatmap.jpg", mimetype="image/jpeg")

# -----------------------------------------------------------------------------
# Run (for local dev)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
