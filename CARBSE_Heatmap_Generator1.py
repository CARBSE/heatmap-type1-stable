# -*- coding: utf-8 -*-
"""
CARBSE Heatmap Generator (Type 1) – modern UI + stable session
- Single-worker friendly: keeps uploaded CSV per-session on disk and in memory
- Responsive Plotly graph inside a card (no overflow)
- DBT/RH defaults for color scale max (DBT=50, RH=100)
- JPG download via Kaleido
"""
import os
import uuid
from datetime import timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash, make_response
)
from werkzeug.utils import secure_filename

# ───────────────────────────── Config ─────────────────────────────
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")
app.permanent_session_lifetime = timedelta(days=3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
PLOT_ROOT = os.path.join(BASE_DIR, "plots")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(PLOT_ROOT, exist_ok=True)

# In-memory map (single worker only)
SESSIONS = {}  # session_id -> {"file": path}

ALLOWED_EXT = {".csv", ".xlsx", ".xls"}


# ─────────────────────── Utility: parse datetime ───────────────────────
def parse_datetime(series: pd.Series) -> pd.Series:
    """Try to parse a 'Timestamp' robustly."""
    # First attempt: general
    dt = pd.to_datetime(series, errors="coerce")
    # If mostly NaT, try dayfirst
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt


# ─────────────────────── Utility: colorscale ───────────────────────
def carbse_colorscale():
    """
    Custom 8-color gradient:
    White → Navy → Blue → Cyan → Yellow → Red → Dark Red → Black
    """
    return [
        [0.00, "#FFFFFF"],  # White
        [0.14, "#001f3f"],  # Navy blue
        [0.28, "#0074D9"],  # Blue
        [0.42, "#7FDBFF"],  # Cyan
        [0.56, "#FFDC00"],  # Yellow
        [0.70, "#FF4136"],  # Red
        [0.84, "#85144b"],  # Dark red
        [1.00, "#000000"],  # Black
    ]


# ─────────────────────── Helpers: session & file ───────────────────────
def get_or_create_session_id(resp=None):
    sid = request.cookies.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        if resp is None:
            resp = make_response()
        resp.set_cookie("session_id", sid, max_age=60 * 60 * 24 * 3, samesite="Lax")
    return sid, resp


def session_upload_dir(session_id: str) -> str:
    path = os.path.join(UPLOAD_ROOT, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def find_saved_file(session_id: str) -> str | None:
    # Prefer memory map
    info = SESSIONS.get(session_id)
    if info and os.path.exists(info["file"]):
        return info["file"]
    # Fallback to disk (survives code reload in same pod)
    d = session_upload_dir(session_id)
    candidates = [os.path.join(d, "data.csv"), os.path.join(d, "data.xlsx")]
    for c in candidates:
        if os.path.exists(c):
            # rehydrate memory map
            SESSIONS[session_id] = {"file": c}
            return c
    return None


# ───────────────────────────── Routes ─────────────────────────────
@app.route("/", methods=["GET"])
def home():
    resp = make_response()
    session_id, resp = get_or_create_session_id(resp)
    csv_path = find_saved_file(session_id)

    table_html = None
    cols = []
    filename = None

    if csv_path and os.path.exists(csv_path):
        filename = os.path.basename(csv_path)
        try:
            if csv_path.lower().endswith(".csv"):
                df = pd.read_csv(csv_path)
            else:
                df = pd.read_excel(csv_path)
            cols = df.columns.tolist()
            # Preview table (first 30 rows)
            table_html = df.head(30).to_html(
                classes="table table-sm table-striped table-hover align-middle",
                index=False,
                border=0,
                justify="center",
            )
        except Exception:
            # If read fails, force user to re-upload
            table_html = None
            cols = []
            filename = None

    return render_template(
        "heatmap_app_00.html",
        session_id=session_id,
        filename=filename,
        columns=cols,
        preview_table=table_html,
        graph_html=None,
        image_url=None,
        message=None,
    ), 200, resp.headers


@app.route("/upload", methods=["POST"])
def upload():
    resp = make_response(redirect(url_for("home")))
    session_id, resp = get_or_create_session_id(resp)

    if "file" not in request.files:
        flash("Please choose a file.", "warning")
        return resp

    f = request.files["file"]
    if not f or f.filename.strip() == "":
        flash("Please choose a file.", "warning")
        return resp

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        flash("Only CSV/XLSX files are allowed.", "danger")
        return resp

    # Save under uploads/<session_id>/data.ext
    dst_dir = session_upload_dir(session_id)
    if ext == ".csv":
        dst = os.path.join(dst_dir, "data.csv")
    else:
        dst = os.path.join(dst_dir, "data.xlsx")

    f.save(dst)
    SESSIONS[session_id] = {"file": dst}
    flash("File uploaded successfully. Now pick options and generate the heatmap.", "success")
    return resp


@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    # Session & file
    session_id = request.cookies.get("session_id")
    if not session_id:
        flash("Session expired. Please upload the file again.", "warning")
        return redirect(url_for("home"))

    csv_path = find_saved_file(session_id)
    if not csv_path:
        flash("No uploaded file found. Please upload your CSV/XLSX first.", "warning")
        return redirect(url_for("home"))

    # Parameter
    parameter = request.form.get("parameter", "DBT").strip()
    if parameter not in ("DBT", "RH"):
        parameter = "DBT"

    # Optional overrides
    try:
        vmin = float(request.form.get("scale_min", "").strip()) if request.form.get("scale_min") else None
    except ValueError:
        vmin = None
    try:
        vmax = float(request.form.get("scale_max", "").strip()) if request.form.get("scale_max") else None
    except ValueError:
        vmax = None

    # Defaults based on parameter
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 50.0 if parameter == "DBT" else 100.0

    # Read data
    if csv_path.lower().endswith(".csv"):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_excel(csv_path)

    # Expect columns: Timestamp, DBT, RH
    if "Timestamp" not in df.columns:
        flash("Column 'Timestamp' not found in the uploaded file.", "danger")
        return redirect(url_for("home"))
    if parameter not in df.columns:
        flash(f"Column '{parameter}' not found in the uploaded file.", "danger")
        return redirect(url_for("home"))

    # Clean & transform
    df = df.copy()
    dt = parse_datetime(df["Timestamp"])
    df = df.loc[~dt.isna()].copy()
    df["__date__"] = dt.dt.date
    df["__hour__"] = dt.dt.hour
    df[parameter] = pd.to_numeric(df[parameter], errors="coerce")

    # Pivot: y=hour (0..23), x=date, z=mean(parameter)
    pivot = df.pivot_table(index="__hour__", columns="__date__", values=parameter, aggfunc="mean")
    pivot = pivot.sort_index(axis=0)  # hours ascending
    pivot = pivot.sort_index(axis=1)  # dates ascending

    x_labels = [d.strftime("%Y-%m-%d") for d in pivot.columns]
    y_labels = list(pivot.index)

    # Build figure
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            zmin=vmin,
            zmax=vmax,
            colorscale=carbse_colorscale(),
            colorbar=dict(title=parameter),
            hovertemplate=("Date: %{x}<br>Hour: %{y}<br>" + parameter + ": %{z:.2f}<extra></extra>"),
        )
    )

    fig.update_layout(
        title=f"Single-City Heatmap • {parameter}",
        margin=dict(l=40, r=20, t=60, b=60),
        autosize=True,
        xaxis=dict(tickangle=-45, automargin=True),
        yaxis=dict(title="Hour (0–23)", automargin=True),
        template="plotly_white",
    )

    # Render responsive HTML snippet
    graph_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displaylogo": False, "responsive": True}
    )

    # Save JPG for download
    jpg_path = os.path.join(PLOT_ROOT, f"{session_id}.jpg")
    try:
        fig.write_image(jpg_path, format="jpg", scale=2, width=1400, height=600)
        image_url = url_for("download_jpg", session_id=session_id)
    except Exception:
        # Kaleido might fail for odd fonts; still show chart
        image_url = None

    # Prepare table & columns again for the page
    try:
        preview_table = df[["Timestamp", parameter]].head(30).to_html(
            classes="table table-sm table-striped table-hover align-middle",
            index=False, border=0, justify="center",
        )
        cols = df.columns.tolist()
        filename = os.path.basename(csv_path)
    except Exception:
        preview_table = None
        cols, filename = [], None

    return render_template(
        "heatmap_app_00.html",
        session_id=session_id,
        filename=filename,
        columns=cols,
        preview_table=preview_table,
        graph_html=graph_html,
        image_url=image_url,
        message=None,
        default_vmin=vmin,
        default_vmax=vmax,
        selected_param=parameter,
    )


@app.route("/download/jpg/<session_id>", methods=["GET"])
def download_jpg(session_id):
    jpg_path = os.path.join(PLOT_ROOT, f"{session_id}.jpg")
    if not os.path.exists(jpg_path):
        flash("No plot image available. Please generate a heatmap first.", "warning")
        return redirect(url_for("home"))
    return send_file(jpg_path, mimetype="image/jpeg", as_attachment=True, download_name="heatmap.jpg")


@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200


# ───────────────────────────── Main ─────────────────────────────
if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
