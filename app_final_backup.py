# ============================================================
# HeatShield AI — Kochi Urban Heat Intelligence
# AI-Powered Urban Heat Intelligence & Cooling Optimizer
# ============================================================

from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import tifffile
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HeatShield AI | Kochi",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "kochi_heat_data.csv"
MODEL_FILE = BASE_DIR / "kochi_heat_model.pkl"
RASTER_FILE = BASE_DIR / "Kochi_HeatShield_Raster.tif"
METADATA_FILE = BASE_DIR / "model_metadata.json"


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = ["NDVI", "NDBI", "NDWI"]

# Display/QC limits.
# These are used to suppress obvious raster artifacts when
# presenting surface-temperature information.
MIN_VALID_LST = 10.0
MAX_VALID_LST = 55.0

RISK_LABELS = ["Low", "Moderate", "High", "Extreme"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .section-spacer {
        height: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE.name}")

    df = pd.read_csv(DATA_FILE)

    required = ["LST", "NDVI", "NDBI", "NDWI"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Training data is missing columns: {', '.join(missing)}"
        )

    return df


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_FILE.name}")

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    if not METADATA_FILE.exists():
        return {}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_raster():
    if not RASTER_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RASTER_FILE.name}")

    with tifffile.TiffFile(RASTER_FILE) as tif:
        arr = tif.asarray()

        tags = tif.pages[0].tags

        pixel_scale_tag = tags.get("ModelPixelScaleTag")
        tiepoint_tag = tags.get("ModelTiepointTag")

        pixel_scale = (
            tuple(pixel_scale_tag.value)
            if pixel_scale_tag is not None
            else None
        )

        tiepoint = (
            tuple(tiepoint_tag.value)
            if tiepoint_tag is not None
            else None
        )

    arr = np.asarray(arr, dtype=np.float32)

    # Expected raster:
    # (4, rows, cols)
    #
    # If it comes as:
    # (rows, cols, 4)
    # convert it.
    if arr.ndim == 3 and arr.shape[0] == 4:
        bands = arr

    elif arr.ndim == 3 and arr.shape[-1] == 4:
        bands = np.moveaxis(arr, -1, 0)

    else:
        raise ValueError(
            f"Unexpected raster shape: {arr.shape}. "
            "Expected four bands."
        )

    return bands, pixel_scale, tiepoint


# ============================================================
# SAFE INITIALIZATION
# ============================================================

try:
    df = load_training_data()
    model = load_model()
    metadata = load_metadata()
    raster_bands, pixel_scale, tiepoint = load_raster()

except Exception as e:
    st.error("HeatShield AI could not load the project files.")
    st.exception(e)
    st.stop()


# ============================================================
# RASTER BANDS
# ============================================================

lst_raster = raster_bands[0]
ndvi_raster = raster_bands[1]
ndbi_raster = raster_bands[2]
ndwi_raster = raster_bands[3]

rows, cols = lst_raster.shape


# ============================================================
# GEOGRAPHIC COORDINATES
# ============================================================

def get_geo_coordinates(rows, cols, pixel_scale, tiepoint):
    """
    Derive longitude/latitude arrays from GeoTIFF tags.

    The exported raster uses:
      ModelPixelScaleTag
      ModelTiepointTag

    Pixel centers are used for plotting.
    """

    if pixel_scale is not None and tiepoint is not None:
        pixel_width = abs(float(pixel_scale[0]))
        pixel_height = abs(float(pixel_scale[1]))

        origin_lon = float(tiepoint[3])
        origin_lat = float(tiepoint[4])

        lons = (
            origin_lon
            + (np.arange(cols) + 0.5) * pixel_width
        )

        lats = (
            origin_lat
            - (np.arange(rows) + 0.5) * pixel_height
        )

    else:
        # Fallback based on the known Kochi study area.
        lons = np.linspace(76.20, 76.40, cols)
        lats = np.linspace(10.10, 9.85, rows)

    return lons, lats


lons, lats = get_geo_coordinates(
    rows,
    cols,
    pixel_scale,
    tiepoint,
)


# ============================================================
# DATA CLEANING / DISPLAY QC
# ============================================================

def clean_lst_for_display(array):
    """
    Remove invalid/nonphysical temperature values for dashboard
    visualization.

    This does not modify the original raster file.
    """

    result = np.array(array, dtype=np.float32, copy=True)

    invalid = (
        ~np.isfinite(result)
        | (result < MIN_VALID_LST)
        | (result > MAX_VALID_LST)
    )

    result[invalid] = np.nan

    return result


lst_display = clean_lst_for_display(lst_raster)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

@st.cache_data
def calculate_predictions(
    ndvi,
    ndbi,
    ndwi,
):
    """
    Predict LST for the raster from NDVI/NDBI/NDWI.

    Returns a raster-shaped prediction.
    """

    valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndbi)
        & np.isfinite(ndwi)
    )

    prediction = np.full(
        ndvi.shape,
        np.nan,
        dtype=np.float32,
    )

    if np.any(valid):

        feature_matrix = np.column_stack(
            [
                ndvi[valid],
                ndbi[valid],
                ndwi[valid],
            ]
        )

        prediction[valid] = model.predict(feature_matrix)

    return prediction


predicted_raster = calculate_predictions(
    ndvi_raster,
    ndbi_raster,
    ndwi_raster,
)


# ============================================================
# RISK THRESHOLDS
# ============================================================

valid_training_lst = df["LST"].replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()

# Prefer the training-data distribution for risk thresholds.
p75 = float(valid_training_lst.quantile(0.75))
p90 = float(valid_training_lst.quantile(0.90))
p95 = float(valid_training_lst.quantile(0.95))

def classify_risk(temp):
    if not np.isfinite(temp):
        return "Unknown"

    if temp >= p95:
        return "Extreme"

    if temp >= p90:
        return "High"

    if temp >= p75:
        return "Moderate"

    return "Low"


def risk_rank(label):
    mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Extreme": 4,
    }

    return mapping.get(label, 0)


risk_raster = np.full(
    lst_display.shape,
    "",
    dtype=object,
)

finite_mask = np.isfinite(lst_display)

risk_raster[
    finite_mask
] = np.vectorize(classify_risk)(
    lst_display[finite_mask]
)


# ============================================================
# HOTSPOT EXTRACTION
# ============================================================

@st.cache_data
def extract_hotspots(
    lst,
    predicted,
    ndvi,
    ndbi,
    ndwi,
    lons,
    lats,
    number=30,
):
    """
    Extract the hottest valid pixels from the cleaned raster.
    """

    valid = (
        np.isfinite(lst)
        & np.isfinite(predicted)
        & np.isfinite(ndvi)
        & np.isfinite(ndbi)
        & np.isfinite(ndwi)
    )

    if not np.any(valid):
        return pd.DataFrame()

    flat_indices = np.flatnonzero(valid)

    lst_values = lst.ravel()[flat_indices]

    order = np.argsort(lst_values)[::-1][:number]

    selected = flat_indices[order]

    row_idx, col_idx = np.unravel_index(
        selected,
        lst.shape,
    )

    hotspot_df = pd.DataFrame(
        {
            "Hotspot": [
                f"Hotspot #{i + 1}"
                for i in range(len(selected))
            ],
            "Latitude": lats[col_idx * 0 + row_idx],
            "Longitude": lons[col_idx],
            "Observed LST": lst[row_idx, col_idx],
            "AI Estimated LST": predicted[row_idx, col_idx],
            "NDVI": ndvi[row_idx, col_idx],
            "NDBI": ndbi[row_idx, col_idx],
            "NDWI": ndwi[row_idx, col_idx],
        }
    )

    hotspot_df["Thermal Gap"] = (
        hotspot_df["Observed LST"]
        - hotspot_df["AI Estimated LST"]
    )

    hotspot_df["Risk"] = hotspot_df[
        "Observed LST"
    ].apply(classify_risk)

    hotspot_df["Priority Score"] = (
        hotspot_df["Observed LST"].rank(
            pct=True
        ) * 0.50
        +
        (-hotspot_df["NDVI"]).rank(
            pct=True
        ) * 0.15
        +
        hotspot_df["NDBI"].rank(
            pct=True
        ) * 0.20
        +
        (-hotspot_df["NDWI"]).rank(
            pct=True
        ) * 0.15
    )

    hotspot_df["Priority Score"] = (
        hotspot_df["Priority Score"] * 100
    )

    return hotspot_df


hotspots = extract_hotspots(
    lst_display,
    predicted_raster,
    ndvi_raster,
    ndbi_raster,
    ndwi_raster,
    lons,
    lats,
)


# ============================================================
# MODEL METRICS
# ============================================================

r2 = metadata.get("r2", None)
rmse = metadata.get("rmse", None)
mae = metadata.get("mae", None)

model_name = metadata.get(
    "model",
    "Random Forest",
)

training_samples = metadata.get(
    "training_samples",
    len(df),
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌡️ HeatShield AI")
st.sidebar.subheader("Kochi Urban Heat Intelligence")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Command Center",
        "🌡️ Heat Map",
        "🔥 Hotspot Intelligence",
        "🧊 Cooling Simulator",
        "🤖 AI Planning",
        "📊 Model & Data",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Satellite-derived environmental indicators"
)

st.sidebar.caption(
    "Random Forest heat prediction"
)

st.sidebar.caption(
    "Scenario-based cooling planning"
)


# ============================================================
# HEADER FUNCTION
# ============================================================

def render_header(
    title,
    subtitle,
):
    st.title(title)
    st.write(subtitle)
    st.divider()


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    render_header(
        "🏙️ Kochi HeatShield AI",
        "AI-powered urban heat intelligence and cooling strategy support.",
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    valid_lst_values = lst_display[
        np.isfinite(lst_display)
    ]

    if len(valid_lst_values) > 0:

        max_lst = float(np.nanmax(valid_lst_values))
        mean_lst = float(np.nanmean(valid_lst_values))
        high_fraction = float(
            np.mean(valid_lst_values >= p90) * 100
        )

    else:

        max_lst = np.nan
        mean_lst = np.nan
        high_fraction = np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Maximum valid LST",
            f"{max_lst:.2f}°C",
        )

    with c2:
        st.metric(
            "Mean valid LST",
            f"{mean_lst:.2f}°C",
        )

    with c3:
        st.metric(
            "High / Extreme area",
            f"{high_fraction:.1f}%",
        )

    with c4:
        st.metric(
            "Hotspots analyzed",
            str(len(hotspots)),
        )

    st.write("")

    # --------------------------------------------------------
    # DECISION SIGNAL
    # --------------------------------------------------------

    st.info(
        """
        🏙️ **Kochi Heat Action Signal**

        Prioritize locations with extreme surface temperatures,
        low vegetation signals, stronger built-up signals and
        limited water-related signals.

        HeatShield AI compares environmental intervention
        scenarios to support planning discussions.
        """
    )

    st.subheader("🔥 Kochi Heat Risk Overview")

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    map_step = max(1, int(max(rows, cols) / 180))

    map_lst = lst_display[::map_step, ::map_step]

    map_lat = lats[::map_step]
    map_lon = lons[::map_step]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=map_lst,
                x=map_lon,
                y=map_lat,
                colorscale="Turbo",
                colorbar=dict(
                    title="LST °C"
                ),
                hovertemplate=(
                    "Longitude: %{x:.5f}<br>"
                    "Latitude: %{y:.5f}<br>"
                    "LST: %{z:.2f}°C"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        height=620,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Map shows valid satellite-derived land surface temperature. "
        f"Values outside {MIN_VALID_LST:.0f}–{MAX_VALID_LST:.0f}°C "
        "are excluded from visualization as potential data/QA artifacts."
    )

    # --------------------------------------------------------
    # HOTSPOT SUMMARY
    # --------------------------------------------------------

    st.subheader("🚨 Highest-Temperature Locations")

    if not hotspots.empty:

        display_hotspots = hotspots[
            [
                "Hotspot",
                "Latitude",
                "Longitude",
                "Observed LST",
                "Risk",
                "Priority Score",
            ]
        ].copy()

        display_hotspots["Observed LST"] = (
            display_hotspots["Observed LST"]
            .round(2)
        )

        display_hotspots["Priority Score"] = (
            display_hotspots["Priority Score"]
            .round(1)
        )

        st.dataframe(
            display_hotspots.head(10),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader("🧠 How HeatShield AI Works")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 1️⃣ Detect")
        st.write(
            "Satellite-derived land surface temperature "
            "is used to identify spatial heat patterns."
        )

    with b:
        st.markdown("### 2️⃣ Explain")
        st.write(
            "NDVI, NDBI and NDWI provide environmental "
            "signals used by the Random Forest model."
        )

    with c:
        st.markdown("### 3️⃣ Plan")
        st.write(
            "Cooling scenarios are simulated to compare "
            "potential environmental interventions."
        )


# ============================================================
# HEAT MAP
# ============================================================

elif page == "🌡️ Heat Map":

    render_header(
        "🌡️ Heat Map",
        "Explore satellite-derived environmental conditions across the Kochi study area.",
    )

    layer = st.selectbox(
        "Select environmental layer",
        [
            "Land Surface Temperature",
            "NDVI — Vegetation",
            "NDBI — Built-up Signal",
            "NDWI — Water Signal",
            "AI Estimated LST",
            "Thermal Gap",
        ],
    )

    if layer == "Land Surface Temperature":

        data = lst_display
        title = "Observed Land Surface Temperature"
        colorbar = "LST °C"
        colorscale = "Turbo"

    elif layer == "NDVI — Vegetation":

        data = ndvi_raster
        title = "Normalized Difference Vegetation Index"
        colorbar = "NDVI"
        colorscale = "Greens"

    elif layer == "NDBI — Built-up Signal":

        data = ndbi_raster
        title = "Normalized Difference Built-up Index"
        colorbar = "NDBI"
        colorscale = "Oranges"

    elif layer == "NDWI — Water Signal":

        data = ndwi_raster
        title = "Normalized Difference Water Index"
        colorbar = "NDWI"
        colorscale = "Blues"

    elif layer == "AI Estimated LST":

        data = predicted_raster
        title = "Random Forest Estimated LST"
        colorbar = "Predicted LST °C"
        colorscale = "Turbo"

    else:

        data = lst_display - predicted_raster
        title = "Observed minus AI Estimated LST"
        colorbar = "Thermal Gap °C"
        colorscale = "RdBu_r"

    step = max(
        1,
        int(max(rows, cols) / 200),
    )

    data_plot = data[::step, ::step]

    lat_plot = lats[::step]
    lon_plot = lons[::step]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=data_plot,
                x=lon_plot,
                y=lat_plot,
                colorscale=colorscale,
                colorbar=dict(
                    title=colorbar
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=650,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("📖 Layer Interpretation")

    if layer == "Land Surface Temperature":

        st.write(
            "Higher values indicate hotter land surface conditions "
            "in the satellite observation."
        )

    elif layer == "NDVI — Vegetation":

        st.write(
            "Higher NDVI generally represents stronger vegetation "
            "signals. Lower values can indicate sparse vegetation "
            "or built surfaces."
        )

    elif layer == "NDBI — Built-up Signal":

        st.write(
            "Higher NDBI indicates stronger built-up or impervious "
            "surface characteristics."
        )

    elif layer == "NDWI — Water Signal":

        st.write(
            "Higher NDWI indicates stronger water-related spectral "
            "signals."
        )

    elif layer == "AI Estimated LST":

        st.write(
            "This is the Random Forest prediction based on NDVI, "
            "NDBI and NDWI."
        )

    else:

        st.write(
            "Thermal Gap = observed satellite LST minus AI estimated "
            "LST. Large values indicate locations where the observed "
            "temperature differs substantially from the environmental "
            "feature-based prediction."
        )


# ============================================================
# HOTSPOT INTELLIGENCE
# ============================================================

elif page == "🔥 Hotspot Intelligence":

    render_header(
        "🔥 Hotspot Intelligence",
        "Move from a temperature value to an interpretable AI assessment.",
    )

    st.write(
        "This page combines observed satellite temperature with "
        "the environmental conditions surrounding each hotspot."
    )

    if hotspots.empty:

        st.warning(
            "No valid hotspots were detected after raster quality filtering."
        )

    else:

        hotspot_options = hotspots["Hotspot"].tolist()

        selected_name = st.selectbox(
            "Select hotspot",
            hotspot_options,
        )

        selected = hotspots[
            hotspots["Hotspot"] == selected_name
        ].iloc[0]

        observed = float(
            selected["Observed LST"]
        )

        predicted = float(
            selected["AI Estimated LST"]
        )

        gap = float(
            selected["Thermal Gap"]
        )

        risk = selected["Risk"]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Observed LST",
                f"{observed:.2f}°C",
            )
            st.caption(
                "Satellite observation"
            )

        with c2:
            st.metric(
                "AI Estimated LST",
                f"{predicted:.2f}°C",
            )
            st.caption(
                "Model estimate from environmental indicators"
            )

        with c3:
            st.metric(
                "Risk",
                risk,
            )
            st.caption(
                "Training-data percentile classification"
            )

        with c4:
            st.metric(
                "Thermal Gap",
                f"{gap:+.2f}°C",
            )
            st.caption(
                "Observed minus AI estimate"
            )

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if abs(gap) >= 10:

            st.warning(
                f"""
                ⚠️ **Large observation-model difference**

                The satellite observation and model estimate differ
                by approximately **{gap:+.2f}°C**.

                The satellite value is the observed surface-temperature
                signal. The AI value is a prediction from NDVI, NDBI
                and NDWI.

                The difference should therefore be treated as a
                diagnostic signal rather than proof that either value
                represents the "true" temperature.
                """
            )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        st.subheader("📍 Hotspot Location")

        location_df = pd.DataFrame(
            {
                "Measure": [
                    "Latitude",
                    "Longitude",
                    "Priority Score",
                ],
                "Value": [
                    f"{float(selected['Latitude']):.6f}",
                    f"{float(selected['Longitude']):.6f}",
                    f"{float(selected['Priority Score']):.1f}/100",
                ],
            }
        )

        st.dataframe(
            location_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # ENVIRONMENTAL PROFILE
        # ----------------------------------------------------

        st.subheader("🌱 Environmental Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "NDVI",
                f"{float(selected['NDVI']):.3f}",
            )

        with c2:
            st.metric(
                "NDBI",
                f"{float(selected['NDBI']):.3f}",
            )

        with c3:
            st.metric(
                "NDWI",
                f"{float(selected['NDWI']):.3f}",
            )

        # ----------------------------------------------------
        # PROFILE CHART
        # ----------------------------------------------------

        profile_df = pd.DataFrame(
            {
                "Indicator": [
                    "NDVI",
                    "NDBI",
                    "NDWI",
                ],
                "Value": [
                    float(selected["NDVI"]),
                    float(selected["NDBI"]),
                    float(selected["NDWI"]),
                ],
            }
        )

        fig_profile = px.bar(
            profile_df,
            x="Indicator",
            y="Value",
            title="Environmental Indicator Profile",
        )

        fig_profile.update_layout(
            height=400,
        )

        st.plotly_chart(
            fig_profile,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("🧠 AI Interpretation")

        ndvi_value = float(selected["NDVI"])
        ndbi_value = float(selected["NDBI"])
        ndwi_value = float(selected["NDWI"])

        interpretation = []

        if ndvi_value < float(df["NDVI"].median()):
            interpretation.append(
                "The vegetation signal is below the training-data median."
            )
        else:
            interpretation.append(
                "The vegetation signal is at or above the training-data median."
            )

        if ndbi_value > float(df["NDBI"].median()):
            interpretation.append(
                "The built-up signal is above the training-data median."
            )
        else:
            interpretation.append(
                "The built-up signal is at or below the training-data median."
            )

        if ndwi_value < float(df["NDWI"].median()):
            interpretation.append(
                "The water-related signal is below the training-data median."
            )
        else:
            interpretation.append(
                "The water-related signal is at or above the training-data median."
            )

        for item in interpretation:
            st.write(f"• {item}")

        st.caption(
            "These are descriptive model-input signals. They should "
            "not be interpreted as causal proof that one environmental "
            "factor directly produced the observed temperature."
        )


# ============================================================
# COOLING SIMULATOR
# ============================================================

elif page == "🧊 Cooling Simulator":

    render_header(
        "🧊 Cooling Simulator",
        "Test environmental intervention scenarios against the Random Forest model.",
    )

    if hotspots.empty:

        st.warning(
            "No hotspots are available for simulation."
        )

    else:

        hotspot_options = hotspots["Hotspot"].tolist()

        selected_name = st.selectbox(
            "Choose a hotspot",
            hotspot_options,
        )

        selected = hotspots[
            hotspots["Hotspot"] == selected_name
        ].iloc[0]

        base_ndvi = float(selected["NDVI"])
        base_ndbi = float(selected["NDBI"])
        base_ndwi = float(selected["NDWI"])

        baseline_prediction = float(
            model.predict(
                np.array(
                    [
                        [
                            base_ndvi,
                            base_ndbi,
                            base_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        st.subheader("Current Environmental State")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Observed LST",
                f"{float(selected['Observed LST']):.2f}°C",
            )

        with c2:
            st.metric(
                "AI Baseline",
                f"{baseline_prediction:.2f}°C",
            )

        with c3:
            st.metric(
                "NDVI",
                f"{base_ndvi:.3f}",
            )

        with c4:
            st.metric(
                "NDBI",
                f"{base_ndbi:.3f}",
            )

        # ----------------------------------------------------
        # TRAINING DISTRIBUTION TARGETS
        # ----------------------------------------------------

        ndvi_median = float(df["NDVI"].median())
        ndvi_p75 = float(df["NDVI"].quantile(0.75))

        ndbi_median = float(df["NDBI"].median())
        ndbi_p25 = float(df["NDBI"].quantile(0.25))

        ndwi_median = float(df["NDWI"].median())
        ndwi_p75 = float(df["NDWI"].quantile(0.75))

        scenarios = []

        # Baseline
        baseline = baseline_prediction

        scenarios.append(
            {
                "Intervention": "Baseline",
                "Predicted LST": baseline,
                "Change vs Baseline": 0.0,
            }
        )

        # ----------------------------------------------------
        # GREEN INFRASTRUCTURE
        # ----------------------------------------------------

        green_ndvi = max(
            base_ndvi,
            ndvi_p75,
        )

        green_pred = float(
            model.predict(
                np.array(
                    [
                        [
                            green_ndvi,
                            base_ndbi,
                            base_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        scenarios.append(
            {
                "Intervention": "🌳 Green Infrastructure",
                "Predicted LST": green_pred,
                "Change vs Baseline": green_pred - baseline,
            }
        )

        # ----------------------------------------------------
        # COOL ROOFS / LOWER BUILT-UP SIGNAL
        # ----------------------------------------------------

        cool_ndbi = min(
            base_ndbi,
            ndbi_p25,
        )

        cool_pred = float(
            model.predict(
                np.array(
                    [
                        [
                            base_ndvi,
                            cool_ndbi,
                            base_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        scenarios.append(
            {
                "Intervention": "🏠 Cool Roof / Reflective Surface",
                "Predicted LST": cool_pred,
                "Change vs Baseline": cool_pred - baseline,
            }
        )

        # ----------------------------------------------------
        # BLUE INFRASTRUCTURE
        # ----------------------------------------------------

        blue_ndwi = max(
            base_ndwi,
            ndwi_p75,
        )

        blue_pred = float(
            model.predict(
                np.array(
                    [
                        [
                            base_ndvi,
                            base_ndbi,
                            blue_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        scenarios.append(
            {
                "Intervention": "💧 Blue Infrastructure",
                "Predicted LST": blue_pred,
                "Change vs Baseline": blue_pred - baseline,
            }
        )

        # ----------------------------------------------------
        # COMBINED
        # ----------------------------------------------------

        combined_pred = float(
            model.predict(
                np.array(
                    [
                        [
                            green_ndvi,
                            cool_ndbi,
                            blue_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        scenarios.append(
            {
                "Intervention": "🌳💧 Integrated Cooling",
                "Predicted LST": combined_pred,
                "Change vs Baseline": combined_pred - baseline,
            }
        )

        scenario_df = pd.DataFrame(
            scenarios
        )

        scenario_df["Cooling Magnitude"] = (
            baseline
            - scenario_df["Predicted LST"]
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.subheader("Scenario Comparison")

        table_df = scenario_df.copy()

        table_df["Predicted LST"] = (
            table_df["Predicted LST"].round(2)
        )

        table_df["Change vs Baseline"] = (
            table_df["Change vs Baseline"].round(2)
        )

        table_df["Cooling Magnitude"] = (
            table_df["Cooling Magnitude"].round(2)
        )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        fig = px.bar(
            scenario_df,
            x="Intervention",
            y="Predicted LST",
            title="Modeled LST Under Environmental Scenarios",
        )

        fig.add_hline(
            y=baseline,
            line_dash="dash",
            annotation_text="Baseline",
        )

        fig.update_layout(
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.info(
            """
            💡 **Important modeling note**

            These scenarios modify NDVI, NDBI and NDWI inputs and
            then pass them through the trained Random Forest.

            Therefore, the simulated temperature changes are
            **model-based scenario comparisons**, not guaranteed
            real-world cooling values.
            """
        )


# ============================================================
# AI PLANNING
# ============================================================

elif page == "🤖 AI Planning":

    render_header(
        "🤖 AI Planning",
        "Translate hotspot conditions and modeled scenarios into planning signals.",
    )

    if hotspots.empty:

        st.warning(
            "No valid hotspots are available."
        )

    else:

        # ----------------------------------------------------
        # HOTSPOT SELECTION
        # ----------------------------------------------------

        selected_name = st.selectbox(
            "Select planning hotspot",
            hotspots["Hotspot"].tolist(),
        )

        selected = hotspots[
            hotspots["Hotspot"] == selected_name
        ].iloc[0]

        base_ndvi = float(selected["NDVI"])
        base_ndbi = float(selected["NDBI"])
        base_ndwi = float(selected["NDWI"])

        baseline = float(
            model.predict(
                np.array(
                    [
                        [
                            base_ndvi,
                            base_ndbi,
                            base_ndwi,
                        ]
                    ]
                )
            )[0]
        )

        ndvi_target = max(
            base_ndvi,
            float(df["NDVI"].quantile(0.75)),
        )

        ndbi_target = min(
            base_ndbi,
            float(df["NDBI"].quantile(0.25)),
        )

        ndwi_target = max(
            base_ndwi,
            float(df["NDWI"].quantile(0.75)),
        )

        planning_scenarios = []

        scenario_inputs = {
            "🌳 Green Infrastructure": (
                ndvi_target,
                base_ndbi,
                base_ndwi,
            ),
            "🏠 Cool Roof / Reflective Surface": (
                base_ndvi,
                ndbi_target,
                base_ndwi,
            ),
            "💧 Blue Infrastructure": (
                base_ndvi,
                base_ndbi,
                ndwi_target,
            ),
            "🌳💧 Integrated Cooling": (
                ndvi_target,
                ndbi_target,
                ndwi_target,
            ),
        }

        for name, inputs in scenario_inputs.items():

            prediction = float(
                model.predict(
                    np.array(
                        [[
                            inputs[0],
                            inputs[1],
                            inputs[2],
                        ]]
                    )
                )[0]
            )

            planning_scenarios.append(
                {
                    "Intervention": name,
                    "Predicted LST": prediction,
                    "Change vs Baseline": (
                        prediction - baseline
                    ),
                    "Modeled Cooling": (
                        baseline - prediction
                    ),
                }
            )

        planning_df = pd.DataFrame(
            planning_scenarios
        )

        # Largest modeled decrease.
        largest_reduction = planning_df.loc[
            planning_df["Modeled Cooling"].idxmax()
        ]

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        st.success(
            f"""
            🎯 **AI Planning Signal**

            Among the tested scenarios, the intervention producing
            the largest modeled reduction for this hotspot was:

            **{largest_reduction['Intervention']}**

            Modeled temperature change:

            **{float(largest_reduction['Change vs Baseline']):+.2f}°C**

            This is a model-based scenario comparison, not a claim
            that the intervention will produce exactly this cooling
            in the real environment.
            """
        )

        # ----------------------------------------------------
        # HOTSPOT PROFILE
        # ----------------------------------------------------

        st.subheader("📍 Hotspot Planning Profile")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Observed LST",
                f"{float(selected['Observed LST']):.2f}°C",
            )

        with c2:
            st.metric(
                "AI Baseline",
                f"{baseline:.2f}°C",
            )

        with c3:
            st.metric(
                "Risk",
                selected["Risk"],
            )

        with c4:
            st.metric(
                "Priority Score",
                f"{float(selected['Priority Score']):.1f}/100",
            )

        # ----------------------------------------------------
        # ACTION SIGNALS
        # ----------------------------------------------------

        st.subheader("🧭 Planning Signals")

        if base_ndvi < float(df["NDVI"].median()):

            st.warning(
                "🌳 Vegetation signal is relatively low. "
                "Green infrastructure can be investigated."
            )

        else:

            st.info(
                "🌳 Vegetation signal is not below the "
                "training-data median."
            )

        if base_ndbi > float(df["NDBI"].median()):

            st.warning(
                "🏙️ Built-up signal is relatively high. "
                "Reflective surfaces or cool-roof strategies "
                "can be investigated."
            )

        else:

            st.info(
                "🏙️ Built-up signal is not above the "
                "training-data median."
            )

        if base_ndwi < float(df["NDWI"].median()):

            st.warning(
                "💧 Water-related signal is relatively low. "
                "Blue/blue-green infrastructure can be investigated."
            )

        else:

            st.info(
                "💧 Water-related signal is not below the "
                "training-data median."
            )

        # ----------------------------------------------------
        # SCENARIO TABLE
        # ----------------------------------------------------

        st.subheader("📊 Scenario Planning Matrix")

        display_df = planning_df.copy()

        display_df["Predicted LST"] = (
            display_df["Predicted LST"]
            .round(2)
        )

        display_df["Change vs Baseline"] = (
            display_df["Change vs Baseline"]
            .round(2)
        )

        display_df["Modeled Cooling"] = (
            display_df["Modeled Cooling"]
            .round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = display_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download scenario comparison",
            data=csv_data,
            file_name="kochi_heatshield_scenarios.csv",
            mime="text/csv",
        )

        st.caption(
            "Planning signals are heuristic/model-based and should "
            "be combined with field validation, engineering feasibility, "
            "cost, land availability and community considerations."
        )


# ============================================================
# MODEL & DATA
# ============================================================

elif page == "📊 Model & Data":

    render_header(
        "📊 Model & Data",
        "Technical details, validation metrics and data provenance.",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Model")

    c1, c2, c3 = st.columns(3)

    with c1:

        if r2 is not None:
            st.metric(
                "R²",
                f"{float(r2):.3f}",
            )
        else:
            st.metric(
                "R²",
                "N/A",
            )

    with c2:

        if rmse is not None:
            st.metric(
                "RMSE",
                f"{float(rmse):.3f}°C",
            )
        else:
            st.metric(
                "RMSE",
                "N/A",
            )

    with c3:

        if mae is not None:
            st.metric(
                "MAE",
                f"{float(mae):.3f}°C",
            )
        else:
            st.metric(
                "MAE",
                "N/A",
            )

    st.write("")

    model_info = pd.DataFrame(
        {
            "Property": [
                "Model",
                "Target",
                "Features",
                "Training samples",
                "Estimators",
                "Maximum depth",
            ],
            "Value": [
                model_name,
                metadata.get("target", "LST"),
                ", ".join(
                    metadata.get(
                        "features",
                        FEATURES,
                    )
                ),
                metadata.get(
                    "training_samples",
                    len(df),
                ),
                metadata.get(
                    "n_estimators",
                    "200",
                ),
                metadata.get(
                    "max_depth",
                    "15",
                ),
            ],
        }
    )

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🧠 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.feature_importances_,
            }
        )

        importance_df["Importance %"] = (
            importance_df["Importance"] * 100
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False,
        )

        fig_importance = px.bar(
            importance_df,
            x="Feature",
            y="Importance %",
            title="Random Forest Feature Importance",
        )

        fig_importance.update_layout(
            height=450,
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
        )

        st.caption(
            "Feature importance indicates predictive contribution "
            "within this trained Random Forest. It does not establish "
            "causality."
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("🗂️ Training Dataset")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )

    with c2:
        st.metric(
            "Mean LST",
            f"{df['LST'].mean():.2f}°C",
        )

    with c3:
        st.metric(
            "P90",
            f"{p90:.2f}°C",
        )

    with c4:
        st.metric(
            "P95",
            f"{p95:.2f}°C",
        )

    st.dataframe(
        df.describe().round(3),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    st.subheader("📈 Observed vs AI Prediction")

    sample_size = min(
        2000,
        len(df),
    )

    sample_df = df.sample(
        sample_size,
        random_state=42,
    ).copy()

    sample_predictions = model.predict(
        sample_df[FEATURES]
    )

    validation_df = pd.DataFrame(
        {
            "Observed LST": sample_df["LST"].values,
            "AI Predicted LST": sample_predictions,
        }
    )

    fig_validation = px.scatter(
        validation_df,
        x="Observed LST",
        y="AI Predicted LST",
        title="Observed vs Random Forest Predicted LST",
        opacity=0.55,
    )

    min_value = float(
        min(
            validation_df["Observed LST"].min(),
            validation_df["AI Predicted LST"].min(),
        )
    )

    max_value = float(
        max(
            validation_df["Observed LST"].max(),
            validation_df["AI Predicted LST"].max(),
        )
    )

    fig_validation.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line_dash="dash",
    )

    fig_validation.update_layout(
        height=550,
    )

    st.plotly_chart(
        fig_validation,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # DATA PROVENANCE
    # --------------------------------------------------------

    st.subheader("🛰️ Data & Methodology")

    st.markdown(
        """
        **Satellite-derived variables**

        - **LST** — Land Surface Temperature
        - **NDVI** — vegetation signal
        - **NDBI** — built-up signal
        - **NDWI** — water-related signal

        **Machine learning**

        A Random Forest regression model estimates LST from
        NDVI, NDBI and NDWI.

        **Risk classification**

        Risk categories are based on percentiles of the training
        LST distribution rather than universal health thresholds.

        **Scenario planning**

        Environmental indicators are modified within the model
        to compare hypothetical intervention scenarios.
        """
    )

    st.warning(
        """
        ⚠️ **Model limitation**

        The model identifies statistical relationships in the
        available dataset. Feature importance is not causal evidence,
        and scenario outputs should not be interpreted as guaranteed
        real-world cooling.

        Field measurements, urban morphology, costs, feasibility,
        land availability and community requirements should be
        considered before implementation.
        """
    )


# ============================================================
# GLOBAL FOOTER
# ============================================================

st.divider()

st.caption(
    "🌡️ HeatShield AI — AI-Powered Urban Heat Intelligence & Cooling Optimizer"
)

st.caption(
    "Kochi Urban Heat Study • Satellite + Machine Learning + Scenario Planning"
)