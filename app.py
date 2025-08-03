# app.py

import os
from pathlib import Path
import io
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import csv
import altair as alt
from PIL import Image
import streamlit.components.v1 as components
import urllib.parse  # Add this for URL decoding
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import time
import uuid
import shutil
import numpy as np

# Rasterio imports for GeoTIFF support
try:
    import rasterio
    from rasterio.plot import show
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    st.warning(
        "Rasterio not available. GeoTIFF files will be treated as regular images."
    )

# GeoPandas imports for GPKG support
try:
    import geopandas as gpd
    import streamlit.components.v1 as components
    import folium
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("GeoPandas not available. GPKG files will not be supported.")

from utils.minio_client import MinioClient

# how long before we consider a preview stale (in seconds)
CACHE_TTL = 60 * 60  # 1 hour

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
TEXT_EXT = {".txt"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
GEOTIFF_EXT = {".tif", ".tiff"}  # Separate category for potential GeoTIFFs
GPKG_EXT = {".gpkg"}  # GeoPackage files
PDF_EXT = {".pdf"}
MAX_PREVIEW_SIZE = 100 * 1024 * 1024  # 100 MB for general files
GPKG_MAX_SIZE = (
    1000 * 1024 * 1024
) # 1000 MB for GeoPackages (larger due to spatial data)
GEOTIFF_MAX_SIZE = (
    2750 * 1024 * 1024
)  # 2750 MB for GeoTIFFs (larger due to spatial data)
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks for streaming

# Page setup
st.set_page_config(
    page_title="Resource Previewer", layout="wide", initial_sidebar_state="collapsed"
)
bin_count = 20


@contextmanager
def stream_file_from_minio(mc: MinioClient, s3_path):
    """Stream file from MinIO without downloading entirely to disk first"""
    temp_file = None
    response = None

    try:
        # Create a temporary file for streaming
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(s3_path).suffix.lower()
        )

        # Stream the file in chunks
        try:
            bucket, object = mc._parse_s3_path(s3_path)
            response = mc.client.get_object(bucket, object)
        except Exception as e:
            # Handle MinIO specific errors gracefully
            error_msg = str(e).lower()
            if "nosuchkey" in error_msg or "not found" in error_msg:
                raise FileNotFoundError("The requested file was not found")
            elif "access denied" in error_msg or "forbidden" in error_msg:
                raise PermissionError("Access denied to the requested file")
            elif "invalid" in error_msg and "credentials" in error_msg:
                raise PermissionError("Invalid credentials provided")
            else:
                raise ConnectionError(
                    f"Failed to access file from storage: {error_msg}"
                )

        try:
            while True:
                chunk = response.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_file.flush()
            yield Path(temp_file.name)
        finally:
            if response:
                try:
                    response.close()
                    response.release_conn()
                except:
                    pass
            if temp_file:
                temp_file.close()
    finally:
        # Clean up temp file
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass


def display_gpkg(file_path):
    """Display GeoPackage with geopandas and folium (proxy-safe version)"""
    if not GEOPANDAS_AVAILABLE:
        st.error("GeoPandas is required to display GPKG files but is not installed.")
        return False

    try:
        # List all layers in the GPKG
        import fiona

        layers = fiona.listlayers(str(file_path))
        st.subheader("GeoPackage Information")
        st.write(f"**Layers found:** {len(layers)}")

        if not layers:
            st.warning("No layers found in this GeoPackage file.")
            return False

        # Layer selection
        if len(layers) > 1:
            selected_layer = st.selectbox("Select layer to display:", layers)
        else:
            selected_layer = layers[0]
            st.write(f"**Layer:** {selected_layer}")

        # Read the selected layer
        with st.spinner(f"Loading layer: {selected_layer}..."):
            gdf = gpd.read_file(file_path, layer=selected_layer)

        # Display basic info
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Features:** {len(gdf)}")
            st.write(
                f"**Geometry Type:** {gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else 'Unknown'}"
            )
            st.write(f"**CRS:** {gdf.crs}")
            st.write(f"**Columns:** {len(gdf.columns)}")

        with col2:
            if len(gdf) > 0:
                bounds = gdf.total_bounds
                st.write(f"**Bounds:**")
                st.write(f"  Min X: {bounds[0]:.6f}")
                st.write(f"  Min Y: {bounds[1]:.6f}")
                st.write(f"  Max X: {bounds[2]:.6f}")
                st.write(f"  Max Y: {bounds[3]:.6f}")

        # Show first few rows as table
        if st.checkbox("Show attribute table"):
            if len(gdf) > 0:
                # Convert geometry to WKT for display
                display_df = gdf.copy()
                display_df["geometry"] = display_df["geometry"].apply(
                    lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
                )
                st.dataframe(display_df.head(100), use_container_width=True)
            else:
                st.write("No features to display")

        # Create interactive map
        if len(gdf) > 0:
            st.subheader("Interactive Map")

            # Reproject to WGS84 for folium
            if gdf.crs != "EPSG:4326":
                gdf_wgs84 = gdf.to_crs("EPSG:4326")
            else:
                gdf_wgs84 = gdf

            # Create folium map
            bounds = gdf_wgs84.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=10,
                prefer_canvas=True  # Better performance for proxy environments
            )

            # Add alternative tile layers that work better behind proxies
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)

            # Add features to map
            max_features = st.slider(
                "Max features to display on map",
                1,
                min(1000, len(gdf_wgs84)),
                min(100, len(gdf_wgs84)),
            )

            # Sample data if too many features
            if len(gdf_wgs84) > max_features:
                display_gdf = gdf_wgs84.sample(max_features)
                st.info(
                    f"Showing {max_features} randomly sampled features out of {len(gdf_wgs84)} total"
                )
            else:
                display_gdf = gdf_wgs84

            # Add to map based on geometry type
            geom_type = display_gdf.geometry.geom_type.iloc[0]

            if geom_type in ["Point", "MultiPoint"]:
                # Use MarkerCluster for better performance with many points
                from folium.plugins import MarkerCluster
                
                if len(display_gdf) > 50:
                    marker_cluster = MarkerCluster().add_to(m)
                    parent_map = marker_cluster
                else:
                    parent_map = m

                # Add points
                for idx, row in display_gdf.iterrows():
                    # Create popup text
                    popup_lines = [f"<b>Feature {idx}</b>"]
                    
                    # Add attribute information
                    for col in row.index:
                        if col != "geometry" and pd.notna(row[col]):
                            value = str(row[col])
                            if len(value) > 50:
                                value = value[:47] + "..."
                            popup_lines.append(f"<b>{col}:</b> {value}")
                    
                    popup_text = "<br>".join(popup_lines[:8])  # Limit to 8 lines
                    
                    # Get coordinates
                    if geom_type == "Point":
                        coords = [row.geometry.y, row.geometry.x]
                    else:  # MultiPoint
                        coords = [row.geometry.centroid.y, row.geometry.centroid.x]

                    folium.Marker(
                        location=coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"Feature {idx}"
                    ).add_to(parent_map)

            else:
                # Add polygons/lines with enhanced styling and popups
                def style_function(feature):
                    return {
                        "fillColor": "blue",
                        "color": "darkblue",
                        "weight": 2,
                        "fillOpacity": 0.6,
                        "opacity": 0.8
                    }

                def highlight_function(feature):
                    return {
                        "fillColor": "red",
                        "color": "red",
                        "weight": 3,
                        "fillOpacity": 0.8,
                        "opacity": 1.0
                    }

                # Get fields for popup (exclude geometry)
                popup_fields = [col for col in display_gdf.columns if col != "geometry"][:5]
                tooltip_fields = popup_fields[:3]

                folium.GeoJson(
                    display_gdf,
                    style_function=style_function,
                    highlight_function=highlight_function,
                    popup=folium.GeoJsonPopup(
                        fields=popup_fields,
                        aliases=[col.title() for col in popup_fields],
                        localize=True,
                        max_width=300
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=tooltip_fields,
                        aliases=[col.title() for col in tooltip_fields],
                        localize=True
                    )
                ).add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            # Fit map to bounds with some padding
            southwest = [bounds[1] - 0.01, bounds[0] - 0.01]
            northeast = [bounds[3] + 0.01, bounds[2] + 0.01]
            m.fit_bounds([southwest, northeast])

            # Display map using HTML component (proxy-safe)
            map_html = m._repr_html_()
            components.html(map_html, height=600, scrolling=True)

        # Statistics for numeric columns
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and st.checkbox("Show statistics for numeric columns"):
            st.subheader("Numeric Column Statistics")
            st.dataframe(gdf[numeric_cols].describe(), use_container_width=True)

        return True

    except Exception as e:
        st.error(f"Unable to process this GeoPackage file: {str(e)}")
        return False


def is_geotiff(file_path):
    """Check if a TIFF file is a GeoTIFF"""
    if not RASTERIO_AVAILABLE:
        return False

    try:
        with rasterio.open(file_path) as src:
            return src.crs is not None
    except Exception:
        return False


def display_geotiff(file_path):
    """Display GeoTIFF with rasterio"""
    try:
        with rasterio.open(file_path) as src:
            # Display basic info
            st.subheader("GeoTIFF Information")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Dimensions:** {src.width} x {src.height}")
                st.write(f"**Bands:** {src.count}")
                st.write(f"**Data Type:** {src.dtypes[0]}")
                st.write(f"**CRS:** {src.crs}")

            with col2:
                bounds = src.bounds
                st.write(f"**Bounds:**")
                st.write(f"  Left: {bounds.left:.6f}")
                st.write(f"  Bottom: {bounds.bottom:.6f}")
                st.write(f"  Right: {bounds.right:.6f}")
                st.write(f"  Top: {bounds.top:.6f}")

            # Band selection for multi-band images
            if src.count > 1:
                selected_bands = st.multiselect(
                    "Select bands to display (RGB order)",
                    options=list(range(1, src.count + 1)),
                    default=list(
                        range(1, min(4, src.count + 1))
                    ),  # Default to first 3 bands
                    max_selections=3,
                )
            else:
                selected_bands = [1]

            # Resampling for large images
            max_display_size = st.sidebar.slider(
                "Max display dimension", 512, 2048, 1024
            )

            # Calculate resampling factor
            factor = max(
                src.width / max_display_size, src.height / max_display_size, 1.0
            )

            if factor > 1:
                st.info(f"Image resampled by factor of {factor:.2f} for display")
                out_width = int(src.width / factor)
                out_height = int(src.height / factor)
            else:
                out_width = src.width
                out_height = src.height

            # Read and display the data
            if len(selected_bands) == 1:
                # Single band - display as grayscale
                with st.spinner("Processing single band visualization..."):
                    band_data = src.read(
                        selected_bands[0],
                        out_shape=(out_height, out_width),
                        resampling=Resampling.bilinear,
                    )

                    fig, ax = plt.subplots(figsize=(12, 8))
                    im = ax.imshow(band_data, cmap="gray")
                    ax.set_title(f"Band {selected_bands[0]}")
                    ax.axis("off")
                    plt.colorbar(im)
                    st.pyplot(fig)
                    plt.close()

            elif len(selected_bands) >= 2:
                # Multi-band - create RGB composite
                with st.spinner("Processing RGB composite visualization..."):
                    bands_data = []
                    for band in selected_bands[:3]:  # Take max 3 bands
                        band_data = src.read(
                            band,
                            out_shape=(out_height, out_width),
                            resampling=Resampling.bilinear,
                        )
                        bands_data.append(band_data)

                    # Stack bands and normalize for display
                    if len(bands_data) == 2:
                        # Add a dummy third band for RGB
                        bands_data.append(np.zeros_like(bands_data[0]))

                    rgb_data = np.stack(bands_data, axis=2)

                    # Normalize to 0-255 range
                    for i in range(rgb_data.shape[2]):
                        band = rgb_data[:, :, i]
                        if band.max() > band.min():
                            band_min, band_max = np.percentile(band, [2, 98])
                            rgb_data[:, :, i] = np.clip(
                                (band - band_min) / (band_max - band_min) * 255, 0, 255
                            )

                    rgb_data = rgb_data.astype(np.uint8)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(rgb_data)
                    ax.set_title(f"RGB Composite (Bands: {selected_bands[:3]})")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

            # Statistics
            if st.checkbox("Show band statistics"):
                st.subheader("Band Statistics")
                with st.spinner("Calculating band statistics..."):
                    stats_data = []
                    for i in range(1, src.count + 1):
                        try:
                            band_data = src.read(i)
                            stats_data.append(
                                {
                                    "Band": i,
                                    "Min": float(band_data.min()),
                                    "Max": float(band_data.max()),
                                    "Mean": float(band_data.mean()),
                                    "Std": float(band_data.std()),
                                    "NoData": src.nodata,
                                }
                            )
                        except Exception:
                            # Skip bands that can't be processed
                            continue

                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        st.warning("Unable to calculate statistics for this GeoTIFF")

    except Exception as e:
        st.error(
            "Unable to process this GeoTIFF file. It may be corrupted or in an unsupported format."
        )
        # Fallback to regular image display
        return False

    return True


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
            # Always use tab for .tsv
            if ext == ".tsv":
                sep = "\t"
            # If we see more semicolons than commas in the sample, assume ;)
            elif sample.count(";") > sample.count(","):
                sep = ";"
            else:
                try:
                    sep = csv.Sniffer().sniff(sample).delimiter
                except csv.Error:
                    sep = "," if ext == ".csv" else "\t"

        engine = "python" if nrows else "pyarrow"
        df = pd.read_csv(
            path,
            sep=sep,
            engine=engine,
            encoding="latin1",
            on_bad_lines="skip",
            skiprows=range(1, skiprows + 1),
            nrows=nrows,
        )

        # (Optional) drop any purely-empty or unnamed "Unnamed:" columns
        df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed.*")]

        # (Optional) clean up whitespace in your headers
        df.columns = df.columns.str.strip()

        return df

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
            y=alt.Y("count()", axis=None),
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
access_key = q.get("access_key")
secret_key = q.get("secret_key")
session_token = q.get("session_token")
s3_endpoint = q.get("s3_endpoint")
s3_path_raw = q.get("s3_path")

# Credentials check
if not all([access_key, secret_key, s3_endpoint, s3_path_raw]):
    st.error(
        "Please provide `access_key`, `secret_key`, `s3_endpoint` and `s3_path` in the URL query params."
    )
    st.stop()

# URL decode the s3_path parameter
if s3_path_raw:
    s3_path = urllib.parse.unquote(s3_path_raw)
else:
    s3_path = None

# Init MinIO client
mc = MinioClient(
    endpoint=s3_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    session_token=session_token,
    secure=False,
)

# Parse bucket/object and size
try:
    with st.spinner("Accessing file information..."):
        size = mc.get_size(s3_path=s3_path)
        original_filename = os.path.basename(s3_path)

        # Show file info
        st.info(f"**File:** {original_filename} ({size / (1024*1024):.1f} MB)")

except FileNotFoundError:
    st.error("❌ **File not found**. Please check the file path and try again.")
    st.stop()
except PermissionError:
    st.error("❌ **Access denied**. Please check your credentials and permissions.")
    st.stop()
except ConnectionError as e:
    st.error(f"❌ **Connection error**: Unable to connect to storage service.")
    st.stop()
except Exception:
    st.error("❌ **Error accessing file**. Please check your parameters and try again.")
    st.stop()

# Determine extension
suffix = Path(original_filename).suffix.lower()
supported = IMAGE_EXT | TEXT_EXT | PDF_EXT | TABULAR_EXT | GEOTIFF_EXT | GPKG_EXT

# Unsupported format
if suffix not in supported:
    st.warning(f"**Unsupported format**: `{suffix}`")
    try:
        file_bytes = mc.get_object(s3_path=s3_path)
        st.download_button(
            "Download file",
            file_bytes,
            file_name=original_filename,
            mime="application/octet-stream",
        )
    except Exception:
        st.error("❌ **Unable to prepare file for download**")
    st.stop()

# ── Preview ──────────────────────────────────────────────────────────────────

# Handle TIFF files (could be regular images or GeoTIFFs)
if suffix in GEOTIFF_EXT:
    # Use larger size limit for GeoTIFFs
    max_size_for_type = GEOTIFF_MAX_SIZE

    # Check size against GeoTIFF limit
    if size > max_size_for_type:
        st.warning(
            f"**TIFF file too large to preview (>{max_size_for_type/1024**2:.0f} MB)**"
        )
        try:
            file_bytes = mc.get_object(s3_path=s3_path)
            st.download_button(
                "Download file",
                file_bytes,
                file_name=original_filename,
                mime="application/octet-stream",
            )
        except Exception:
            st.error("❌ **Unable to prepare file for download**")
        st.stop()

    with st.spinner("Loading TIFF file..."):
        try:
            with stream_file_from_minio(mc, s3_path) as temp_path:
                # Check if it's a GeoTIFF
                if is_geotiff(temp_path):
                    st.success("GeoTIFF detected!")
                    display_geotiff(temp_path)
                else:
                    # Treat as regular image
                    try:
                        img = Image.open(temp_path)
                        if img.mode not in {"RGB", "RGBA", "L"}:
                            img = img.convert("RGB")
                        st.image(
                            img,
                            caption=f"{img.format} • {img.size[0]}×{img.size[1]} px",
                        )
                    except Exception:
                        st.error(
                            "Unable to display this TIFF file. It may be corrupted or in an unsupported format."
                        )
        except FileNotFoundError:
            st.error("❌ **File not found during processing**")
        except PermissionError:
            st.error("❌ **Access denied during processing**")
        except ConnectionError:
            st.error("❌ **Connection error during file processing**")
        except Exception:
            st.error(
                "❌ **Unable to process TIFF file**. The file may be corrupted or in an unsupported format."
            )
elif suffix in GPKG_EXT:
    # Check size against general limit for GPKG files
    if size > GPKG_MAX_SIZE:
        st.warning(
            f"**GPKG file too large to preview (>{MAX_PREVIEW_SIZE/1024**2:.0f} MB)**"
        )
        try:
            file_bytes = mc.get_object(s3_path=s3_path)
            st.download_button(
                "Download file",
                file_bytes,
                file_name=original_filename,
                mime="application/octet-stream",
            )
        except Exception:
            st.error("❌ **Unable to prepare file for download**")
        st.stop()

    with st.spinner("Loading GeoPackage..."):
        try:
            with stream_file_from_minio(mc, s3_path) as temp_path:
                st.success("GeoPackage detected!")
                display_gpkg(temp_path)
        except FileNotFoundError:
            st.error("❌ **File not found during processing**")
        except PermissionError:
            st.error("❌ **Access denied during processing**")
        except ConnectionError:
            st.error("❌ **Connection error during file processing**")
        except Exception as e:
            st.write(f"**Debug - GPKG error: {type(e).__name__}: {str(e)}**")
            st.error(
                "❌ **Unable to process GeoPackage file**. The file may be corrupted or in an unsupported format."
            )
elif suffix in IMAGE_EXT:
    # Check size against general limit
    if size > MAX_PREVIEW_SIZE:
        st.warning(
            f"**Image too large to preview (>{MAX_PREVIEW_SIZE/1024**2:.0f} MB)**"
        )
        try:
            file_bytes = mc.get_object(s3_path=s3_path)
            st.download_button(
                "Download file",
                file_bytes,
                file_name=original_filename,
                mime="application/octet-stream",
            )
        except Exception:
            st.error("❌ **Unable to prepare file for download**")
        st.stop()

    with st.spinner("Loading image..."):
        try:
            with stream_file_from_minio(mc, s3_path) as temp_path:
                try:
                    img = Image.open(temp_path)
                    if img.mode not in {"RGB", "RGBA", "L"}:
                        img = img.convert("RGB")
                    st.image(
                        img, caption=f"{img.format} • {img.size[0]}×{img.size[1]} px"
                    )
                except Exception:
                    st.error(
                        "Unable to display this image. It may be corrupted or in an unsupported format."
                    )
        except FileNotFoundError:
            st.error("❌ **File not found during processing**")
        except PermissionError:
            st.error("❌ **Access denied during processing**")
        except ConnectionError:
            st.error("❌ **Connection error during file processing**")
        except Exception:
            st.error("❌ **Unable to process image file**")

elif suffix in TEXT_EXT:
    # Check size against general limit
    if size > MAX_PREVIEW_SIZE:
        st.warning(
            f"Large text file ({size / (1024*1024):.1f} MB). Showing first part only."
        )
        # Read first chunk only
        try:
            bucket_name, object_name = mc._parse_s3_path(s3_path)
            response = mc.client.get_object(bucket_name, object_name)
            try:
                chunk = response.read(1024 * 1024)  # 1MB
                txt = chunk.decode("utf-8", errors="replace")
                st.text_area("Text content (partial)", txt, height=500)
            finally:
                response.close()
                response.release_conn()
        except Exception:
            st.error("❌ **Unable to read text file**")
    else:
        with st.spinner("Loading text file..."):
            try:
                with stream_file_from_minio(mc, s3_path) as temp_path:
                    try:
                        txt = temp_path.read_text(errors="replace")
                        st.text_area("Text content", txt, height=500)
                    except Exception:
                        st.error(
                            "Unable to decode text file. It may be in an unsupported encoding."
                        )
            except FileNotFoundError:
                st.error("❌ **File not found during processing**")
            except PermissionError:
                st.error("❌ **Access denied during processing**")
            except ConnectionError:
                st.error("❌ **Connection error during file processing**")
            except Exception:
                st.error("❌ **Unable to process text file**")

elif suffix in PDF_EXT:
    # Check size against general limit
    if size > MAX_PREVIEW_SIZE:
        st.warning(f"**PDF too large to preview (>{MAX_PREVIEW_SIZE/1024**2:.0f} MB)**")
        try:
            file_bytes = mc.get_object(s3_path=s3_path)
            st.download_button(
                "Download file",
                file_bytes,
                file_name=original_filename,
                mime="application/octet-stream",
            )
        except Exception:
            st.error("❌ **Unable to prepare file for download**")
        st.stop()

    with st.spinner("Loading PDF..."):
        try:
            with stream_file_from_minio(mc, s3_path) as temp_path:
                try:
                    st.components.v1.html(_pdf_embed(temp_path), height=720)
                except Exception:
                    st.error(
                        "Unable to display this PDF. It may be corrupted or password-protected."
                    )
        except FileNotFoundError:
            st.error("❌ **File not found during processing**")
        except PermissionError:
            st.error("❌ **Access denied during processing**")
        except ConnectionError:
            st.error("❌ **Connection error during file processing**")
        except Exception:
            st.error("❌ **Unable to process PDF file**")

else:  # Must be tabular
    # Check size against general limit for tabular files
    if size > MAX_PREVIEW_SIZE:
        st.warning(
            f"**Tabular file too large to preview (>{MAX_PREVIEW_SIZE/1024**2:.0f} MB)**"
        )
        try:
            file_bytes = mc.get_object(s3_path=s3_path)
            st.download_button(
                "Download file",
                file_bytes,
                file_name=original_filename,
                mime="application/octet-stream",
            )
        except Exception:
            st.error("❌ **Unable to prepare file for download**")
        st.stop()

    with st.spinner("Loading tabular data..."):
        try:
            with stream_file_from_minio(mc, s3_path) as temp_path:
                try:
                    # Sidebar is closed by default
                    PREVIEW_ROWS = st.sidebar.number_input(
                        "Rows per page",
                        min_value=100,
                        max_value=5000,
                        value=1000,
                        step=100,
                    )
                    offset = st.session_state.get("offset", 0)

                    try:
                        df = _read_tabular(
                            temp_path, nrows=PREVIEW_ROWS, skiprows=offset
                        )
                    except Exception:
                        st.error(
                            "Unable to parse this tabular file. It may be corrupted or in an unsupported format."
                        )
                        st.stop()

                    # Filter
                    filt = st.sidebar.text_input("Filter rows (regex)")
                    if filt:
                        try:
                            df = df[
                                df.astype(str)
                                .apply(
                                    lambda r: r.str.contains(filt, regex=True, na=False)
                                )
                                .any(axis=1)
                            ]
                        except Exception:
                            st.warning(
                                "Invalid filter expression. Showing unfiltered data."
                            )

                    cols = df.columns.tolist()
                    if cols:  # Only proceed if we have columns
                        chart_cols = st.columns(len(cols), gap="medium")
                        for c, col_ct in zip(cols, chart_cols):
                            try:
                                chart = _column_chart(df, c, bin_count).configure_axis(
                                    labelFontSize=7, titleFontSize=7
                                )
                                col_ct.altair_chart(chart, use_container_width=True)
                            except Exception:
                                # Skip problematic columns for charting
                                continue

                        st.dataframe(df, use_container_width=True, height=700)

                        try:
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download filtered CSV",
                                csv_bytes,
                                file_name=f"filtered_{temp_path.name}",
                            )
                        except Exception:
                            st.warning("Unable to generate CSV download for this data.")

                        # Pagination - Note: This is simplified since we're streaming
                        if len(df) == PREVIEW_ROWS:  # Assume there might be more data
                            if st.button(
                                f"Load more rows (starting from row {offset + PREVIEW_ROWS + 1})"
                            ):
                                st.session_state.offset = offset + PREVIEW_ROWS
                                st.rerun()
                    else:
                        st.warning("No data columns found in this file.")

                except Exception:
                    st.error("❌ **Unable to process tabular file**")
        except FileNotFoundError:
            st.error("❌ **File not found during processing**")
        except PermissionError:
            st.error("❌ **Access denied during processing**")
        except ConnectionError:
            st.error("❌ **Connection error during file processing**")
        except Exception:
            st.error("❌ **Unable to process tabular file**")

# Always provide download option
st.divider()
try:
    # Get the raw bytes for download (no local_path provided)
    file_bytes = mc.get_object(s3_path=s3_path)
    st.download_button(
        "📥 Download Original File",
        file_bytes,
        file_name=original_filename,
        mime="application/octet-stream",
    )
except Exception:
    st.error("❌ **Unable to prepare file for download**")
