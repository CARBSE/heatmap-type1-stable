# -*- coding: utf-8 -*-
"""
CARBSE Heatmap Generator (Type 1) – full UI, session-safe (single-city)
Flask + Plotly + Pandas + Kaleido
"""

import os
import uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, send_file, jsonify, redirect, url_for
)
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# App & folders
# --------------------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
TEMPLATE_DIR = os.path.join(BASE_DIR, "Templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# Map session_id -> info
SESSION_STORE = {}  # { session_id: {"file": path, "df_head_html": str} }

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def read_csv_safely(csv_path: str) -> pd.DataFrame:
    """Read CSV and try to parse 'Timestamp' robustly."""
    df = pd.read_csv(csv_path)

    # Normalise headers we expect: Timestamp, DBT, RH (case-insensitive OK)
    # We'll keep original names if present, but try canonical aliases.
    col_map = {c.lower(): c for c in df.columns}
    # Mandatory Timestamp
    ts_col = None
    for key in ["timestamp", "time", "date_time", "datetime"]:
        if key in col_map:
            ts_col = col_map[key]
            break
    if ts_col is None:
        # Fall back to the most likely 'Timestamp'
        ts_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]

    # Ensure Timestamp is datetime
    # (remove deprecated infer_datetime_format)
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        try:
            dt = pd.to_datetime(df[ts_col], errors="coerce")
            bad = dt.isna().sum()
            df[ts_col] = dt
        except Exception:
            # Final fallback: try dayfirst
            dt = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
            df[ts_col] = dt

    # Rename to canonical names we’ll use downstream (but keep original available)
    if ts_col != "Timestamp":
        df = df.rename(columns={ts_col: "Timestamp"})
    # Standardise possible variants for DBT / RH
    for want, candidates in {
        "DBT": ["dbt", "drybulb", "dry_bulb", "temp", "temperature", "airtemp", "dry bulb temperature"],
        "RH":  ["rh", "relhum", "relative_humidity", "rel humidity", "humidity"]
    }.items():
        if want not in df.columns:
            for c in df.columns:
                if c.lower() in candidates:
                    df = df.rename(columns={c: want})
                    break

    # Add Hour (0..23) for pivot
    df["Hour"] = df["Timestamp"].dt.hour

    # For safety, sort by time
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def colorscale_from_names(names):
    """Build a Plotly colorscale list from selected colour names."""
    if not names:
        return "Viridis"

    # Base palette
    palette = {
        "white":   "#FFFFFF",
        "navyblue":"#001f4d",
        "blue":    "#1f77b4",
        "cyan":    "#17becf",
        "yellow":  "#ffea00",
        "red":     "#ff4136",
        "darkred": "#8B0000",
        "black":   "#000000",
        "grey":    "#808080",
        "gray":    "#808080",
    }

    chosen = []
    for n in names:
        key = n.strip().lower()
        if key in palette:
            chosen.append(palette[key])

    if len(chosen) < 2:
        return "Viridis"

    # even-spaced [0..1]
    stops = np.linspace(0, 1, len(chosen)).tolist()
    colorscale = [[float(s), c] for s, c in zip(stops, chosen)]
    return colorscale


def x_tickformat_from_choice(choice: str) -> str:
    """Map radio choice to Plotly time tickformat."""
    mapping = {
        "Jan-17":        "%b-%y",
        "01-Jan":        "%d-%b",
        "01-01-17":      "%d-%m-%y",
        "01-January":    "%d-%B",
        "Mon":           "%a",
        "Monday":        "%A",
        "Jan":           "%b",
        "01":            "%m",
        "17":            "%y",
        "2017":          "%Y",
    }
    return mapping.get(choice, "%b-%y")


def x_dtick_from_resolution(reso: str) -> str:
    """Plotly time dtick strings."""
    reso = (reso or "").lower()
    if reso == "days":
        return "D1"
    if reso == "weeks":
        return "D7"    # pseudo-week
    if reso == "months":
        return "M1"
    if reso == "years":
        return "M12"
    return None


def css_dash(style: str) -> str:
    style = (style or "dotted").lower()
    return {"dotted": "dot", "dashed": "dash", "solid": "solid"}.get(style, "dot")


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    # fresh session
    return render_template(
        "heatmap_app_00.html",
        session_id="",
        graph_html="",
        table_html="",
        state=default_state()
    )


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template(
            "heatmap_app_00.html",
            session_id="",
            graph_html="",
            table_html="<div class='text-danger'>Please choose a CSV.</div>",
            state=default_state()
        )

    sid = str(uuid.uuid4())
    sdir = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    fname = secure_filename(f.filename)
    fpath = os.path.join(sdir, fname)
    f.save(fpath)

    df = read_csv_safely(fpath)

    # small table preview
    table_html = df.head(100).to_html(
        classes="table table-sm table-striped table-hover mb-0", index=False, border=0
    )

    SESSION_STORE[sid] = {"file": fpath, "table_html": table_html}

    # Pre-populate scale max based on default variable (DBT)
    st = default_state()
    st["session_id"] = sid

    return render_template(
        "heatmap_app_00.html",
        session_id=sid,
        graph_html="",
        table_html=table_html,
        state=st,
    )


def default_state():
    """Default UI values (used to repopulate form)."""
    return {
        "variable": "DBT",
        "scale_min": "",
        "scale_max": "",
        "img_w": "1200",
        "img_h": "600",

        "missing_color": "grey",
        "range_colors": ["white", "navyblue", "blue", "cyan", "yellow", "red", "darkred", "black"],

        "legend_step": "",
        "x_tickfmt": "Jan-17",
        "x_resolution": "months",

        "title": "",
        "x_label": "Date",
        "y_label": "Hour",
        "legend_label": "",

        "upper_time": "09:00",
        "upper_style": "dotted",
        "upper_color": "black",

        "lower_time": "18:00",
        "lower_style": "dotted",
        "lower_color": "black",

        "show_grid": "yes"
    }


@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    form = default_state()
    # Capture posted values (fall back to defaults)
    session_id = request.form.get("session_id", "").strip()

    # if session missing, go back nicely
    if not session_id or session_id not in SESSION_STORE:
        # Graceful redirect to home with a message (but we keep simple)
        return redirect(url_for("home"))

    # update state
    for key in form.keys():
        if key in request.form:
            val = request.form.getlist(key) if key == "range_colors" else request.form.get(key)
            form[key] = val

    # Load data
    file_path = SESSION_STORE[session_id]["file"]
    df = read_csv_safely(file_path)

    # Select variable
    parameter = (form["variable"] or "DBT").strip()
    if parameter not in df.columns:
        # If missing, try case-insensitive
        matches = [c for c in df.columns if c.lower() == parameter.lower()]
        parameter = matches[0] if matches else "DBT"

    # Build pivot: Hours vs Date
    # We'll plot one cell per (date, hour)
    df["Date"] = df["Timestamp"].dt.date
    pivot = df.pivot_table(index="Hour", columns="Date", values=parameter, aggfunc="mean")
    x_vals = list(pivot.columns)
    y_vals = list(pivot.index)

    # Colour scale
    colorscale = colorscale_from_names(form["range_colors"])

    # Scale min / max
    cmin = None
    cmax = None
    if (form["scale_min"] or "").strip():
        try:
            cmin = float(form["scale_min"])
        except Exception:
            cmin = None
    if (form["scale_max"] or "").strip():
        try:
            cmax = float(form["scale_max"])
        except Exception:
            cmax = None

    # Legend step → explicit ticks
    tickvals = None
    if (form["legend_step"] or "").strip():
        try:
            step = float(form["legend_step"])
            if step > 0:
                vmin = cmin if cmin is not None else np.nanmin(pivot.values)
                vmax = cmax if cmax is not None else np.nanmax(pivot.values)
                tickvals = list(np.arange(vmin, vmax + step, step))
        except Exception:
            pass

    # Hover template (escape %{...} with doubled braces in f-string)
    hovertemplate = f"Date: %{{x}}<br>Hour: %{{y}}<br>{parameter}: %{{z:.2f}}<extra></extra>"

    # Build figure
    heatmap_kwargs = dict(
        z=pivot.values,
        x=x_vals,
        y=y_vals,
        colorscale=colorscale,
        colorbar=dict(title=form["legend_label"] or parameter),
        hovertemplate=hovertemplate,
        hoverongaps=True
    )
    if cmin is not None:
        heatmap_kwargs["zmin"] = cmin
    if cmax is not None:
        heatmap_kwargs["zmax"] = cmax

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    # Gridlines
    show_grid = (form["show_grid"] or "yes").lower() == "yes"
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    # Date tick format & spacing
    tickformat = x_tickformat_from_choice(form["x_tickfmt"])
    fig.update_xaxes(tickformat=tickformat)

    dtick = x_dtick_from_resolution(form["x_resolution"])
    if dtick:
        fig.update_xaxes(dtick=dtick)

    # Upper/lower band lines (times in HH:MM across entire width)
    def parse_hour(hhmm):
        try:
            t = datetime.strptime(hhmm.strip(), "%H:%M")
            return t.hour + t.minute / 60.0
        except Exception:
            return None

    upper_h = parse_hour(form["upper_time"])
    lower_h = parse_hour(form["lower_time"])

    x0 = x_vals[0] if x_vals else None
    x1 = x_vals[-1] if x_vals else None

    if x0 is not None and x1 is not None:
        if upper_h is not None:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=upper_h, y1=upper_h,
                line=dict(color=form["upper_color"], width=1.5, dash=css_dash(form["upper_style"])),
                xref="x", yref="y"
            )
        if lower_h is not None:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=lower_h, y1=lower_h,
                line=dict(color=form["lower_color"], width=1.5, dash=css_dash(form["lower_style"])),
                xref="x", yref="y"
            )

    # Background colour used as "missing colour" (transparent cells let this show)
    bg = form["missing_color"] or "white"

    # Legend ticks
    if tickvals:
        fig.update_traces(colorbar=dict(tickmode="array", tickvals=tickvals, ticktext=[str(t) for t in tickvals]))

    # Layout
    try:
        w = int(form["img_w"])
    except Exception:
        w = 1200
    try:
        h = int(form["img_h"])
    except Exception:
        h = 600

    fig.update_layout(
        title=form["title"],
        xaxis_title=form["x_label"] or "Date",
        yaxis_title=form["y_label"] or "Hour",
        width=w, height=h,
        paper_bgcolor="white",
        plot_bgcolor=bg,
        margin=dict(l=40, r=40, t=60, b=60)
    )

    # Save HTML & JPG for download
    html_path = os.path.join(PLOT_DIR, f"{session_id}.html")
    jpg_path  = os.path.join(PLOT_DIR, f"{session_id}.jpg")

    # HTML inline + CDN
    graph_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # JPG for download
    try:
        import kaleido  # noqa
        fig.write_image(jpg_path, format="jpg", scale=2)
    except Exception:
        # if kaleido not present or fails, we still render HTML
        jpg_path = None

    # Table preview
    table_html = SESSION_STORE[session_id].get("table_html", "")

    # Also stash plot path (optional)
    SESSION_STORE[session_id]["plot_jpg"] = jpg_path

    # Re-render template with state retained
    form["session_id"] = session_id
    return render_template(
        "heatmap_app_00.html",
        session_id=session_id,
        graph_html=graph_html,
        table_html=table_html,
        state=form
    )


@app.route("/download/jpg/<session_id>")
def download_jpg(session_id):
    info = SESSION_STORE.get(session_id)
    if not info:
        return "Invalid session", 404
    jpg_path = info.get("plot_jpg")
    if not jpg_path or not os.path.exists(jpg_path):
        return "No image to download. Generate a heatmap first.", 404
    return send_file(jpg_path, as_attachment=True, download_name="heatmap.jpg")


@app.route("/download/csv/<session_id>")
def download_csv(session_id):
    info = SESSION_STORE.get(session_id)
    if not info:
        return "Invalid session", 404
    csv_path = info["file"]
    return send_file(csv_path, as_attachment=True, download_name=os.path.basename(csv_path))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # local run
    app.run(debug=True, use_reloader=False)
