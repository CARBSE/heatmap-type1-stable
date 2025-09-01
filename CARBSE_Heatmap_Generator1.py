# -*- coding: utf-8 -*-
"""
CARBSE Heatmap Generator (Type 1) - Worker-safe (disk-based sessions)
- Upload CSV once per session (Timestamp, DBT, RH)
- Choose parameter (DBT or RH)
- Auto scale max: DBT=50, RH=100 (can be overridden)
- Plotly heatmap (Date vs Hour)
- JPG download via Kaleido
"""
import os
import uuid
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_file
)
from werkzeug.utils import secure_filename

# ---- App setup -------------------------------------------------------------
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("SECRET_KEY", "please-change-me")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv"}

# Color scale (White → Navy → Blue → Cyan → Yellow → Red → DarkRed → Black)
CARBSE_COLORSCALE = [
    [0.00, "#FFFFFF"],  # White
    [0.14, "#000080"],  # NavyBlue
    [0.28, "#0000FF"],  # Blue
    [0.42, "#00FFFF"],  # Cyan
    [0.57, "#FFFF00"],  # Yellow
    [0.71, "#FF0000"],  # Red
    [0.85, "#8B0000"],  # DarkRed
    [1.00, "#000000"],  # Black
]


# ---- Helpers ---------------------------------------------------------------
def _session_dir(session_id: str) -> str:
    return os.path.join(UPLOAD_DIR, session_id)


def _get_session_csv(session_id: str) -> str | None:
    sdir = _session_dir(session_id)
    if not os.path.isdir(sdir):
        return None
    for name in os.listdir(sdir):
        if name.lower().endswith(".csv"):
            return os.path.join(sdir, name)
    return None


def _allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Try to parse timestamps; auto-detect dayfirst if needed."""
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if dt.isna().mean() > 0.50:
        # Try dayfirst=True if initial parse failed heavily
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    return dt


def _preview_rows(df: pd.DataFrame, max_rows: int = 20) -> list[dict]:
    """Return first rows as list of dicts for template rendering."""
    return df.head(max_rows).to_dict(orient="records")


# ---- Routes ----------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("heatmap_app_00.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Please select a CSV file to upload.", "warning")
        return redirect(url_for("home"))

    if not _allowed_file(f.filename):
        flash("Only CSV files are allowed.", "warning")
        return redirect(url_for("home"))

    # Create unique session folder
    session_id = str(uuid.uuid4())
    sdir = _session_dir(session_id)
    os.makedirs(sdir, exist_ok=True)

    # Save the uploaded file
    fname = secure_filename(f.filename)
    fpath = os.path.join(sdir, fname)
    f.save(fpath)

    # Load to get columns + preview
    try:
        df = pd.read_csv(fpath)
        # Expected columns: "Timestamp", "DBT", "RH" (case-sensitive in your doc)
        columns = df.columns.tolist()
        preview = _preview_rows(df, 20)
    except Exception as e:
        flash(f"Could not read CSV: {e}", "danger")
        return redirect(url_for("home"))

    return render_template(
        "heatmap_app_00.html",
        session_id=session_id,
        filename=fname,
        columns=columns,
        preview_rows=preview,
        # Defaults for UI
        selected_parameter="DBT",
        scale_max_default=50,
        scale_max=50,
        width=1200,
        height=600,
    )


@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    session_id = request.form.get("session_id")
    if not session_id:
        flash("Please upload a file first.", "warning")
        return redirect(url_for("home"))

    csv_path = _get_session_csv(session_id)
    if not csv_path:
        flash("Session expired or file not found. Please upload again.", "warning")
        return redirect(url_for("home"))

    # Read selections (sticky)
    parameter = request.form.get("parameter", "DBT")  # "DBT" or "RH"
    width = int(request.form.get("img_width", 1200) or 1200)
    height = int(request.form.get("img_height", 600) or 600)

    # Auto default max: DBT=50, RH=100, but allow override
    auto_max = 50 if parameter == "DBT" else 100
    scale_max_text = request.form.get("scale_max", str(auto_max)).strip()
    try:
        scale_max = float(scale_max_text)
    except Exception:
        scale_max = float(auto_max)

    # Load CSV fresh each time (worker-safe)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        flash(f"Could not re-load CSV: {e}", "danger")
        return redirect(url_for("home"))

    if "Timestamp" not in df.columns or parameter not in df.columns:
        flash("CSV must contain 'Timestamp' and the selected parameter column.", "danger")
        return redirect(url_for("home"))

    # Parse time, build Date + Hour
    dt = _parse_timestamp(df["Timestamp"])
    df = df.copy()
    df["__Date__"] = dt.dt.date
    df["__Hour__"] = dt.dt.hour

    # Drop rows where timestamp failed to parse
    df = df.dropna(subset=["__Date__", "__Hour__"])

    # Pivot (Hour vs Date)
    # index: Hour (0..23), columns: Date, values: mean of selected parameter
    try:
        pivot = df.pivot_table(
            index="__Hour__", columns="__Date__", values=parameter, aggfunc="mean"
        ).sort_index(axis=1)
    except Exception as e:
        flash(f"Could not build heatmap data: {e}", "danger")
        return redirect(url_for("home"))

    x_vals = [d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d) for d in pivot.columns]
    y_vals = list(pivot.index)

    # Build Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_vals,
            y=y_vals,
            colorscale=CARBSE_COLORSCALE,
            zmin=0,
            zmax=scale_max,
            colorbar=dict(title=parameter),
            hovertemplate="Date: %{x}<br>Hour: %{y}<br>%s: %{z:.2f}<extra></extra>" % parameter,
        )
    )
    fig.update_layout(
        title=f"Heatmap of {parameter}",
        xaxis_title="Date",
        yaxis_title="Hour of Day",
        width=width,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, dtick=1, range=[23, 0])  # Hour top→bottom

    # Embed HTML and also write a JPG for download
    graph_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # Save JPG in the user's session folder
    jpg_path = os.path.join(_session_dir(session_id), "heatmap.jpg")
    try:
        fig.write_image(
            jpg_path,
            format="jpg",
            width=width,
            height=height,
            scale=2,
            engine="kaleido",
        )
    except Exception as e:
        # Non-fatal: allow page to render, user can still see interactive plot
        flash(f"Note: could not render JPG (download will fail) - {e}", "warning")

    # Prepare table preview again (so user doesn't need to re-upload)
    try:
        preview = _preview_rows(df[["Timestamp", parameter]], 20)
        columns = df.columns.tolist()
    except Exception:
        preview = None
        columns = df.columns.tolist()

    return render_template(
        "heatmap_app_00.html",
        session_id=session_id,
        filename=os.path.basename(csv_path),
        columns=columns,
        preview_rows=preview,
        graph_html=graph_html,
        # sticky UI values
        selected_parameter=parameter,
        scale_max_default=(50 if parameter == "DBT" else 100),
        scale_max=scale_max,
        width=width,
        height=height,
    )


@app.route("/download/jpg/<session_id>", methods=["GET"])
def download_jpg(session_id):
    jpg_path = os.path.join(_session_dir(session_id), "heatmap.jpg")
    if not os.path.isfile(jpg_path):
        flash("No JPG available. Please generate the heatmap first.", "warning")
        return redirect(url_for("home"))
    return send_file(jpg_path, as_attachment=True, download_name="heatmap.jpg")


@app.route("/download/csv/<session_id>", methods=["GET"])
def download_csv(session_id):
    csv_path = _get_session_csv(session_id)
    if not csv_path or not os.path.isfile(csv_path):
        flash("Original CSV not found for this session.", "warning")
        return redirect(url_for("home"))
    return send_file(csv_path, as_attachment=True, download_name="original_data.csv")


# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    # Local dev
    app.run(debug=True, use_reloader=False)
