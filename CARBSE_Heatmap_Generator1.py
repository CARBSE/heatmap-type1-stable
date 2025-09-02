import os
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash
)
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# App & folders
# -----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
PLOTS_DIR  = BASE_DIR / "plots"
for d in (UPLOAD_DIR, PLOTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "carbse-heatmap-secret")

# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------
LINE_STYLES = {"dotted": "dot", "dashed": "dash", "solid": "solid"}

DATE_TICKFORMAT = {
    "mon_abbr": "%b",              # Jan
    "mon_abbr_2digit_year": "%b-%y",  # Jan-17
    "mon_dec_2digit_year": "%m-%y",   # 01-17
    "mon_dec": "%m",               # 01
    "month_full": "%B",            # January
    "weekday_abbr": "%a",          # Mon
    "weekday_full": "%A",          # Monday
    "year_2digit": "%y",           # 17
    "year_4digit": "%Y",           # 2017
}

DEFAULT_COLORSCALE = [
    "#ffffff", "#001f3f", "#0050a4", "#00bcd4",
    "#ffee58", "#ff5722", "#8b0000", "#000000"
]

COLOR_NAME_MAP = {
    "White": "#ffffff",
    "Navyblue": "#001f3f",
    "Blue": "#0050a4",
    "Cyan": "#00bcd4",
    "Yellow": "#ffee58",
    "Red": "#ff5722",
    "Darkred": "#8b0000",
    "Black": "#000000",
    "Grey": "#c7c7c7",
    "Gray": "#c7c7c7",
}

def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parsing; accepts day-first or not.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().mean() > 0.5:  # try dayfirst if many fails
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt

def colors_to_scale(chosen_names: List[str]) -> List[str]:
    if not chosen_names:
        return DEFAULT_COLORSCALE
    out = []
    for name in chosen_names:
        out.append(COLOR_NAME_MAP.get(name, name))
    # ensure at least 2 stops
    if len(out) < 2:
        out = out + DEFAULT_COLORSCALE
    return out

def _dtick_from_rep(rep: str) -> Optional[str | int]:
    if rep == "months": return "M1"
    if rep == "quarters": return "M3"
    if rep == "years": return "M12"
    if rep == "weeks": return 7 * 24 * 3600 * 1000  # ms
    if rep == "days": return 24 * 3600 * 1000
    return None

# -----------------------------------------------------------------------------
# Core heatmap builder (with NaN mask layer)
# -----------------------------------------------------------------------------
def build_heatmap(
    df: pd.DataFrame,
    parameter: str,
    zmin: Optional[float],
    zmax: Optional[float],
    colors: List[str],
    nodata_color: str,
    upper_time: Optional[str],
    upper_style: str,
    upper_color: str,
    lower_time: Optional[str],
    lower_style: str,
    lower_color: str,
    legend_step: Optional[float],
    tickformat_key: str,
    x_representation: str,
    title: str,
    xname: str,
    yname: str,
    lname: str,
    width: int,
    height: int,
    base_font: int,
) -> go.Figure:

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

    colorscale = colors_to_scale(colors)

    fig = go.Figure()

    # --- 1) NaN mask background (paints only missing cells) ---
    if pivot.size > 0:
        nan_mask = np.where(np.isnan(pivot.values), 1.0, 0.0)
        fig.add_trace(
            go.Heatmap(
                z=nan_mask,
                x=pivot.columns,
                y=pivot.index,
                zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, nodata_color]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    # --- 2) Main heatmap (on top) ---
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            coloraxis="coloraxis",
            connectgaps=False,
            hovertemplate=f"Date: %{{x}}<br>Hour: %{{y}}<br>{parameter}: %{{z:.2f}}<extra></extra>",
            showscale=True,
        )
    )

    # Colorbar ticks
    cbar_opts = {}
    if legend_step and legend_step > 0:
        cbar_opts.update(dict(tickmode="linear", dtick=legend_step))

    tickformat = DATE_TICKFORMAT.get(tickformat_key, "%b-%y")
    dtick = _dtick_from_rep(x_representation)

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=(lname or parameter), **cbar_opts),
        ),
        title=title or f"Single-City Heatmap • {parameter}",
        xaxis=dict(
            title=xname or "Date",
            type="date",
            tickformat=tickformat,
            dtick=dtick,
            showgrid=False,
        ),
        yaxis=dict(
            title=yname or "Hour",
            tickmode="linear", tick0=0, dtick=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        width=width,
        height=height,
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#FFFFFF",
        font=dict(size=base_font, family="Inter, system-ui, Segoe UI, Roboto, Helvetica, Arial"),
    )

    # Optional band lines
    def hh_to_int(hhmm: Optional[str]) -> Optional[int]:
        if not hhmm:
            return None
        try:
            return int(str(hhmm).split(":")[0])
        except Exception:
            return None

    x0 = pivot.columns.min() if not pivot.empty else None
    x1 = pivot.columns.max() if not pivot.empty else None
    if x0 is not None and x1 is not None:
        up = hh_to_int(upper_time)
        lo = hh_to_int(lower_time)
        if up is not None:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=up, y1=up,
                line=dict(color=upper_color or "black", width=1.6, dash=LINE_STYLES.get(upper_style, "dot")),
                xref="x", yref="y",
            )
        if lo is not None:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=lo, y1=lo,
                line=dict(color=lower_color or "black", width=1.6, dash=LINE_STYLES.get(lower_style, "dot")),
                xref="x", yref="y",
            )
    return fig

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "heatmap_app_00.html",
        file_id=None,
        preview=None,
        columns=[],
        fig_html=None,
        plot_id=None,
        defaults=dict(
            param="DBT",
            scale_min="",
            scale_max="",
            colors=["White", "Navyblue", "Blue", "Cyan", "Yellow", "Red", "Darkred", "Black"],
            nodata_color="Grey",
            upper_time="09:00",
            upper_style="dotted",
            upper_color="black",
            lower_time="18:00",
            lower_style="dotted",
            lower_color="black",
            legend_step="10",
            tickformat="mon_abbr",
            x_representation="months",
            title="",
            xname="Date",
            yname="Hour",
            lname="Legend",
            width="1200",
            height="600",
            base_font="15",
        ),
    )

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a CSV/Excel file.", "warning")
        return redirect(url_for("home"))

    # Persist file with a unique id
    file_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{file_id}.csv"

    # CSV or Excel
    if f.filename.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(f)
    else:
        # Try comma/semicolon
        content = f.read()
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception:
            df = pd.read_csv(pd.io.common.BytesIO(content), sep=";")

    # Save normalized CSV to disk
    df.to_csv(path, index=False)

    preview = df.head(30)
    columns = list(df.columns)

    return render_template(
        "heatmap_app_00.html",
        file_id=file_id,
        preview=preview.to_dict(orient="records"),
        columns=columns,
        fig_html=None,
        plot_id=None,
        defaults=request.form,  # keeps previous choices if any
    )

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    file_id = request.form.get("file_id")
    if not file_id:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("home"))

    path = UPLOAD_DIR / f"{file_id}.csv"
    if not path.exists():
        flash("Uploaded file not found on server. Please upload again.", "danger")
        return redirect(url_for("home"))

    df = pd.read_csv(path)

    # Collect form values
    parameter   = request.form.get("parameter", "DBT")
    scale_min   = request.form.get("scale_min", "").strip()
    scale_max   = request.form.get("scale_max", "").strip()
    zmin = float(scale_min) if scale_min else None
    zmax = float(scale_max) if scale_max else None

    colors      = request.form.getlist("colors")
    nodata_color= COLOR_NAME_MAP.get(request.form.get("nodata_color", "Grey"), "#c7c7c7")

    upper_time  = request.form.get("upper_time", "")
    upper_style = request.form.get("upper_style", "dotted")
    upper_color = request.form.get("upper_color", "black")
    lower_time  = request.form.get("lower_time", "")
    lower_style = request.form.get("lower_style", "dotted")
    lower_color = request.form.get("lower_color", "black")

    legend_step = request.form.get("legend_step", "").strip()
    legend_step = float(legend_step) if legend_step else None

    tickformat_key   = request.form.get("tickformat", "mon_abbr")
    x_representation = request.form.get("x_representation", "months")

    title   = request.form.get("title", "")
    xname   = request.form.get("xname", "Date")
    yname   = request.form.get("yname", "Hour")
    lname   = request.form.get("lname", "Legend")

    width   = int(request.form.get("width", "1200") or 1200)
    height  = int(request.form.get("height", "600") or 600)
    base_font = int(request.form.get("base_font", "15") or 15)

    # Build figure
    fig = build_heatmap(
        df=df,
        parameter=parameter,
        zmin=zmin, zmax=zmax,
        colors=colors,
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
        title=title, xname=xname, yname=yname, lname=lname,
        width=width, height=height, base_font=base_font,
    )

    # Save JPG for download
    plot_id = str(uuid.uuid4())
    jpg_path = PLOTS_DIR / f"{plot_id}.jpg"
    fig.write_image(jpg_path, format="jpg", width=width, height=height, scale=1)

    # Embed plot
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Preview table again
    preview = df.head(30)
    columns = list(df.columns)

    return render_template(
        "heatmap_app_00.html",
        file_id=file_id,
        preview=preview.to_dict(orient="records"),
        columns=columns,
        fig_html=fig_html,
        plot_id=plot_id,
        defaults=request.form,
    )

@app.route("/download/jpg/<plot_id>", methods=["GET"])
def download_jpg(plot_id: str):
    path = PLOTS_DIR / f"{plot_id}.jpg"
    if not path.exists():
        flash("Image not found. Please generate again.", "warning")
        return redirect(url_for("home"))
    return send_file(path, mimetype="image/jpeg", as_attachment=True, download_name=f"heatmap_{plot_id}.jpg")

# -----------------------------------------------------------------------------
# Gunicorn entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
