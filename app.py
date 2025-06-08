# app.py

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import csv
import altair as alt
from PIL import Image
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import time
import uuid
import shutil


from utils.minio_client import MinioClient


# how long before we consider a preview stale (in seconds)
CACHE_TTL = 60 * 60   # 1 hour

# root for all previews
TMP_DIR = Path(tempfile.gettempdir()) / "streamlit_previews"
TMP_DIR.mkdir(parents=True, exist_ok=True)

now = time.time()
for child in TMP_DIR.iterdir():
    try:
        if now - child.stat().st_mtime > CACHE_TTL:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    except Exception:
        pass

# per-session folder
if "preview_dir" not in st.session_state:
    session_id = str(uuid.uuid4())
    session_folder = TMP_DIR / session_id
    session_folder.mkdir()
    st.session_state.preview_dir = session_folder



TABULAR_EXT = {".csv", ".tsv", ".json", ".xlsx"}
TEXT_EXT    = {".txt"}
IMAGE_EXT   = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
PDF_EXT     = {".pdf"}
MAX_PREVIEW_SIZE = 40 * 1024 * 1024  # 40 MB

# Page setup
st.set_page_config(page_title="Resource Previewer", layout="wide", initial_sidebar_state="collapsed")
bin_count = 20

def _is_jsonl(path):
    with open(path, "rb") as f:
        head = f.read(128).splitlines()
    return len(head) > 1 and head[0].strip().endswith(b"}")

@st.cache_data
def _read_tabular(path, nrows=None, skiprows=0):
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        with open(path, "r", encoding="latin1", errors="replace") as f:
            sample = f.read(4096)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except csv.Error:
                sep = "," if ext == ".csv" else "\t"
        engine = "python" if nrows else "pyarrow"
        return pd.read_csv(path, sep=sep, engine=engine,
                           encoding="latin1", on_bad_lines="skip",
                           skiprows=range(1, skiprows+1), nrows=nrows)
    if ext == ".json":
        return pd.read_json(path, lines=_is_jsonl(path))
    if ext == ".xlsx":
        return pd.read_excel(path, nrows=nrows, skiprows=skiprows, engine="openpyxl")
    raise ValueError("Unsupported tabular format")

def _column_chart(df, col, maxbins):
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        data = df[[col]].dropna()
        x_enc = col
    else:
        lengths = series.fillna("").astype(str).str.len()
        data = pd.DataFrame({f"{col}_len": lengths})
        x_enc = f"{col}_len"
    return (
        alt.Chart(data)
           .mark_bar()
           .encode(
               x=alt.X(x_enc, bin=alt.Bin(maxbins=maxbins), axis=None),
               y=alt.Y('count()', axis=None),
           )
           .properties(height=80, width=100)
    )

def _pdf_embed(path):
    b = path.read_bytes()
    return f"""
    <div style="overflow-y:auto;height:600px;border:1px solid #ddd;padding:10px;">
      {pdf_viewer(b, annotations=[{{}}])}
    </div>"""

# ── Read URL params ──────────────────────────────────────────────────────────
q = st.query_params
access_key   = q.get("access_key")
secret_key   = q.get("secret_key")
session_token= q.get("session_token")
s3_endpoint  = q.get("s3_endpoint")
s3_path      = q.get("s3_path")

# Credentials check
if not all([access_key, secret_key, s3_endpoint, s3_path]):
    st.error("Please provide `access_key`, `secret_key`, `s3_endpoint` and `s3_path` in the URL query params.")
    st.stop()

# Init MinIO client
mc = MinioClient(
    endpoint=s3_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    session_token=session_token,
    secure=os.getenv("MINIO_INSECURE", "false").lower() == "false"
)

# Parse bucket/object and size
size = mc.get_size(s3_path=s3_path)
original_filename = os.path.basename(s3_path)
local_path = st.session_state.preview_dir / original_filename

# If too big to preview
if size > MAX_PREVIEW_SIZE:
    st.warning(f"**Too large to preview (>{MAX_PREVIEW_SIZE/1024**2:.0f} MB)**")
    st.download_button(
        "Download file",
        mc.get_object(s3_path=s3_path),
        file_name=original_filename,
        mime="application/octet-stream"
    )
    st.stop()

mc.get_object(s3_path=s3_path, local_path=str(local_path))

# Determine extension
suffix = local_path.suffix.lower()
supported = (IMAGE_EXT | TEXT_EXT | PDF_EXT | TABULAR_EXT)

# Unsupported format
if suffix not in supported:
    st.warning(f"**Unsupported format**: `{suffix}`")
    st.download_button(
        "Download file",
        data=local_path.read_bytes(),
        file_name=local_path.name,
        mime="application/octet-stream"
    )
    st.stop()

# ── Preview ──────────────────────────────────────────────────────────────────
if suffix in IMAGE_EXT:
    img = Image.open(local_path)
    if img.mode not in {"RGB","RGBA","L"}:
        img = img.convert("RGB")
    st.image(img, caption=f"{img.format} • {img.size[0]}×{img.size[1]} px")

elif suffix in TEXT_EXT:
    txt = local_path.read_text(errors="replace")
    st.text_area("Text content", txt, height=500)

elif suffix in PDF_EXT:
    st.components.v1.html(_pdf_embed(local_path), height=720)

else:  # Must be tabular
    #Sidebar is closed by default
    
    PREVIEW_ROWS = st.sidebar.number_input(
        "Rows per page", min_value=100, max_value=5000, value=1000, step=100
    )
    offset = st.session_state.get("offset", 0)
    df = _read_tabular(local_path, nrows=PREVIEW_ROWS, skiprows=offset)

    # Filter
    filt = st.sidebar.text_input("Filter rows (regex)")
    if filt:
        df = df[df.astype(str)
                 .apply(lambda r: r.str.contains(filt, regex=True, na=False))
                 .any(axis=1)]
    
    cols = df.columns.tolist()
    chart_cols = st.columns(len(cols), gap="medium")  # Increased gap between columns
    for c, col_ct in zip(cols, chart_cols):
        chart = _column_chart(df, c, bin_count).configure_axis(
            labelFontSize=7,  # Adjust font size for better readability
            titleFontSize=7
        )
        col_ct.altair_chart(chart, use_container_width=True)

    st.dataframe(df, use_container_width=True, height=700)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv_bytes, file_name=f"filtered_{local_path.name}")

    # Pagination
    total = sum(1 for _ in open(local_path, "rb"))
    if offset + PREVIEW_ROWS < total:
        if st.button(f"Load more rows ({offset+1}–{offset+PREVIEW_ROWS})"):
            st.session_state.offset = offset + PREVIEW_ROWS
            st.rerun()

