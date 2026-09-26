import os
import json
import numpy as np
import pandas as pd
import joblib
import tifffile
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# HEATSHIELD AI — KOCHI
# AI-Powered Urban Heat Intelligence & Cooling Optimizer
# ============================================================

st.set_page_config(
    page_title="HeatShield AI | Kochi",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "kochi_heat_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kochi_heat_model.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "model_metadata.json")
RASTER_FILE = os.path.join(BASE_DIR, "Kochi_HeatShield_Raster.tif")


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = ["NDVI", "NDBI", "NDWI"]

RISK_ORDER = [
    "Low",
    "Moderate",
    "High",
    "Extreme"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: #f7f8fa;
        border: 1px solid #e6e8eb;
        min-height: 125px;
    }

    .metric-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #252936;
    }

    .metric-sub {
        font-size: 13px;
        color: #777;
        margin-top: 5px;
    }

    .decision-box {
        padding: 22px;
        border-radius: 16px;
        background: #f5f7fa;
        border: 1px solid #dfe3e8;
        margin-bottom: 20px;
    }

    .decision-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .decision-text {
        font-size: 15px;
        line-height: 1.6;
        color: #444;
    }

    .warning-box {
        padding: 16px;
        border-radius: 12px;
        background: #fff8e6;
        border: 1px solid #f0d27a;
        margin: 12px 0;
    }

    .info-box {
        padding: 16px;
        border-radius: 12px;
        background: #f1f6ff;
        border: 1px solid #c9d9f5;
        margin: 12px 0;
    }

    .success-box {
        padding: 16px;
        border-radius: 12px;
        background: #f1f8f3;
        border: 1px solid #c8e2cf;
        margin: 12px 0;
    }

    .small-note {
        font-size: 12px;
        color: #777;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


@st.cache_data
def load_raster():

    with tifffile.TiffFile(RASTER_FILE) as tif:
        page = tif.pages[0]

        arr = page.asarray()

        tags = page.tags

        scale_tag = tags.get("ModelPixelScaleTag")
        tie_tag = tags.get("ModelTiepointTag")

        if scale_tag is not None:
            scale = scale_tag.value
            pixel_width = float(scale[0])
            pixel_height = float(scale[1])
        else:
            pixel_width = 0.00026949458523585647
            pixel_height = 0.00026949458523585647

        if tie_tag is not None:
            tie = tie_tag.value
            origin_lon = float(tie[3])
            origin_lat = float(tie[4])
        else:
            origin_lon = 76.19986347002366
            origin_lat = 10.10011806546943

    # Handle both possible TIFF shapes:
    # (bands, rows, cols)
    # or (rows, cols, bands)

    if arr.ndim != 3:
        raise ValueError(
            f"Unexpected raster shape: {arr.shape}. "
            "Expected a 3-dimensional raster."
        )

    if arr.shape[0] == 4:
        raster = arr.astype(np.float32)
    elif arr.shape[2] == 4:
        raster = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    else:
        raise ValueError(
            f"Could not identify 4 raster bands. Shape: {arr.shape}"
        )

    lst = raster[0]
    ndvi = raster[1]
    ndbi = raster[2]
    ndwi = raster[3]

    rows, cols = lst.shape

    # Pixel-center coordinates
    lon = origin_lon + (np.arange(cols) + 0.5) * pixel_width

    # GeoTIFF y-axis normally goes downward while latitude decreases.
    lat = origin_lat - (np.arange(rows) + 0.5) * pixel_height

    return (
        lst,
        ndvi,
        ndbi,
        ndwi,
        lon,
        lat,
        pixel_width,
        pixel_height
    )


def classify_risk(temp, p90, p95):

    if temp >= p95:
        return "Extreme"

    if temp >= p90:
        return "High"

    median_threshold = (p90 + p95) / 2

    if temp >= median_threshold:
        return "Moderate"

    return "Low"


def risk_numeric(risk):

    mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Extreme": 4
    }

    return mapping.get(risk, 0)


def safe_float(value, default=0.0):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:
        return default


def metric_card(title, value, subtitle=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_valid_mask(lst, ndvi, ndbi, ndwi):

    return (
        np.isfinite(lst)
        & np.isfinite(ndvi)
        & np.isfinite(ndbi)
        & np.isfinite(ndwi)
    )


def get_hotspot_pixels(
    lst,
    ndvi,
    ndbi,
    ndwi,
    lon,
    lat,
    percentile=95,
    max_points=500
):

    mask = get_valid_mask(lst, ndvi, ndbi, ndwi)

    threshold = np.nanpercentile(lst[mask], percentile)

    rows, cols = np.where(mask & (lst >= threshold))

    if len(rows) == 0:
        return pd.DataFrame()

    values = lst[rows, cols]

    order = np.argsort(values)[::-1]

    rows = rows[order]
    cols = cols[order]

    if len(rows) > max_points:
        rows = rows[:max_points]
        cols = cols[:max_points]

    hotspot_df = pd.DataFrame(
        {
            "Latitude": lat[rows],
            "Longitude": lon[cols],
            "LST_C": lst[rows, cols],
            "NDVI": ndvi[rows, cols],
            "NDBI": ndbi[rows, cols],
            "NDWI": ndwi[rows, cols]
        }
    )

    return hotspot_df


def predict_pixel(model, ndvi, ndbi, ndwi):

    X = pd.DataFrame(
        [{
            "NDVI": ndvi,
            "NDBI": ndbi,
            "NDWI": ndwi
        }]
    )

    prediction = model.predict(X)[0]

    return float(prediction)


def intervention_diagnosis(ndvi, ndbi, ndwi):

    messages = []

    # Vegetation
    if ndvi < 0.20:
        messages.append(
            (
                "Vegetation",
                "Low",
                "Limited vegetation signal suggests an opportunity "
                "for tree canopy, parks or green corridors."
            )
        )

    elif ndvi < 0.40:
        messages.append(
            (
                "Vegetation",
                "Moderate",
                "Vegetation is present but additional canopy could "
                "provide further shading and evapotranspirative cooling."
            )
        )

    else:
        messages.append(
            (
                "Vegetation",
                "Higher",
                "The area already shows a relatively stronger "
                "vegetation signal."
            )
        )

    # Built-up
    if ndbi > 0.15:
        messages.append(
            (
                "Built-up signal",
                "High",
                "Higher built-up signal indicates potential value "
                "from reflective surfaces and shade interventions."
            )
        )

    elif ndbi > 0:
        messages.append(
            (
                "Built-up signal",
                "Moderate",
                "The area has a moderate built-up signature."
            )
        )

    else:
        messages.append(
            (
                "Built-up signal",
                "Lower",
                "The built-up index is relatively low for this location."
            )
        )

    # Water
    if ndwi < -0.15:
        messages.append(
            (
                "Water signal",
                "Low",
                "Low NDWI indicates limited surface-water signal "
                "at this pixel."
            )
        )

    elif ndwi < 0:
        messages.append(
            (
                "Water signal",
                "Moderate",
                "Some water-related signal is present, but not dominant."
            )
        )

    else:
        messages.append(
            (
                "Water signal",
                "Higher",
                "The location has a comparatively stronger water signal."
            )
        )

    return messages


def calculate_scenario(
    model,
    base_ndvi,
    base_ndbi,
    base_ndwi,
    ndvi_target=None,
    ndbi_target=None,
    ndwi_target=None
):

    new_ndvi = base_ndvi
    new_ndbi = base_ndbi
    new_ndwi = base_ndwi

    if ndvi_target is not None:
        new_ndvi = max(base_ndvi, ndvi_target)

    if ndbi_target is not None:
        new_ndbi = min(base_ndbi, ndbi_target)

    if ndwi_target is not None:
        new_ndwi = max(base_ndwi, ndwi_target)

    prediction = predict_pixel(
        model,
        new_ndvi,
        new_ndbi,
        new_ndwi
    )

    return {
        "NDVI": new_ndvi,
        "NDBI": new_ndbi,
        "NDWI": new_ndwi,
        "Predicted LST": prediction
    }


def build_scenarios(
    model,
    base_ndvi,
    base_ndbi,
    base_ndwi,
    ndvi_p75,
    ndbi_p25,
    ndwi_p75
):

    baseline = predict_pixel(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    scenarios = []

    # Green infrastructure
    green = calculate_scenario(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        ndvi_target=ndvi_p75
    )

    scenarios.append(
        {
            "Intervention": "🌳 Green Infrastructure",
            "Description": "Increase vegetation signal toward the local 75th percentile.",
            "Predicted LST": green["Predicted LST"],
            "Change vs Baseline": green["Predicted LST"] - baseline
        }
    )

    # Cool / reflective surfaces
    cool = calculate_scenario(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        ndbi_target=ndbi_p25
    )

    scenarios.append(
        {
            "Intervention": "🏙️ Cool / Reflective Surfaces",
            "Description": "Reduce built-up signal toward the local 25th percentile.",
            "Predicted LST": cool["Predicted LST"],
            "Change vs Baseline": cool["Predicted LST"] - baseline
        }
    )

    # Blue infrastructure
    blue = calculate_scenario(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        ndwi_target=ndwi_p75
    )

    scenarios.append(
        {
            "Intervention": "💧 Blue Infrastructure",
            "Description": "Increase water-related signal toward the local 75th percentile.",
            "Predicted LST": blue["Predicted LST"],
            "Change vs Baseline": blue["Predicted LST"] - baseline
        }
    )

    # Combined
    combined = calculate_scenario(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        ndvi_target=ndvi_p75,
        ndbi_target=ndbi_p25,
        ndwi_target=ndwi_p75
    )

    scenarios.append(
        {
            "Intervention": "🌳🏙️💧 Combined Strategy",
            "Description": "Apply all three modeled environmental changes together.",
            "Predicted LST": combined["Predicted LST"],
            "Change vs Baseline": combined["Predicted LST"] - baseline
        }
    )

    return pd.DataFrame(scenarios), baseline


def priority_score(risk, ndvi, ndbi, ndwi, observed_lst, p90, p95):

    score = 0

    # Thermal risk
    if observed_lst >= p95:
        score += 40
    elif observed_lst >= p90:
        score += 30
    else:
        score += 15

    # Low vegetation
    if ndvi < 0.20:
        score += 25
    elif ndvi < 0.40:
        score += 15

    # High built-up
    if ndbi > 0.15:
        score += 20
    elif ndbi > 0:
        score += 10

    # Low water signal
    if ndwi < -0.15:
        score += 15
    elif ndwi < 0:
        score += 8

    return min(score, 100)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_training_data()
    model = load_model()
    metadata = load_metadata()

    (
        lst,
        ndvi_raster,
        ndbi_raster,
        ndwi_raster,
        lon,
        lat,
        pixel_width,
        pixel_height
    ) = load_raster()

except Exception as e:

    st.error("HeatShield AI could not load the required project files.")

    st.code(str(e))

    st.info(
        "Make sure app.py, kochi_heat_data.csv, "
        "kochi_heat_model.pkl, model_metadata.json and "
        "Kochi_HeatShield_Raster.tif are in the same folder."
    )

    st.stop()


# ============================================================
# MODEL / THRESHOLD INFORMATION
# ============================================================

p90 = float(np.nanpercentile(df["LST"], 90))
p95 = float(np.nanpercentile(df["LST"], 95))

ndvi_p25 = float(np.nanpercentile(df["NDVI"], 25))
ndvi_p75 = float(np.nanpercentile(df["NDVI"], 75))

ndbi_p25 = float(np.nanpercentile(df["NDBI"], 25))
ndbi_p75 = float(np.nanpercentile(df["NDBI"], 75))

ndwi_p25 = float(np.nanpercentile(df["NDWI"], 25))
ndwi_p75 = float(np.nanpercentile(df["NDWI"], 75))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    # 🌡️ HeatShield AI

    **Kochi Urban Heat Intelligence**

    ---
    """
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Command Center",
        "🌡️ Heat Map",
        "🔥 Hotspot Intelligence",
        "🧊 Cooling Simulator",
        "🤖 AI Planning",
        "📊 Model & Data"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Satellite-derived environmental indicators**

    **Random Forest heat prediction**

    **AI-assisted cooling planning**
    """
)


# ============================================================
# PAGE 1 — COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    st.title("🏠 Kochi Heat Command Center")

    st.markdown(
        """
        ### From satellite observations to actionable cooling decisions

        HeatShield AI combines satellite-derived land surface temperature,
        vegetation, built-up and water indicators with a Random Forest model
        to identify and interpret urban heat hotspots.
        """
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    valid_mask = get_valid_mask(
        lst,
        ndvi_raster,
        ndbi_raster,
        ndwi_raster
    )

    valid_lst = lst[valid_mask]

    extreme_count = int(np.sum(valid_lst >= p95))
    high_count = int(np.sum(valid_lst >= p90))

    mean_lst = float(np.nanmean(valid_lst))
    max_lst = float(np.nanmax(valid_lst))

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Mean LST",
            f"{mean_lst:.1f}°C",
            "Study-area satellite surface temperature"
        )

    with c2:
        metric_card(
            "Maximum LST",
            f"{max_lst:.1f}°C",
            "Highest valid raster observation"
        )

    with c3:
        metric_card(
            "Extreme Pixels",
            f"{extreme_count:,}",
            "Pixels at or above local 95th percentile"
        )

    with c4:
        metric_card(
            "Model R²",
            f"{float(metadata.get('r2', 0)):.3f}",
            "Held-out test performance"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DECISION BOX
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="decision-box">
            <div class="decision-title">
                🏙️ Kochi Heat Action Signal
            </div>
            <div class="decision-text">
                Prioritize locations with extreme surface temperatures,
                low vegetation signals, stronger built-up signals and
                limited water-related signals. HeatShield AI can then
                compare modeled environmental intervention scenarios
                before implementation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # HEAT MAP
    # --------------------------------------------------------

    st.subheader("🔥 Where is the heat concentrated?")

    # Downsample for browser performance
    step = max(1, int(max(lst.shape) / 180))

    map_lst = lst[::step, ::step]
    map_lat = lat[::step]
    map_lon = lon[::step]

    heat_df = pd.DataFrame(
        {
            "LST": map_lst.ravel(),
            "Latitude": np.repeat(map_lat, len(map_lon)),
            "Longitude": np.tile(map_lon, len(map_lat))
        }
    )

    heat_df = heat_df[np.isfinite(heat_df["LST"])]

    fig = px.density_map(
        heat_df,
        lat="Latitude",
        lon="Longitude",
        z="LST",
        radius=8,
        center={
            "lat": float(np.nanmean(lat)),
            "lon": float(np.nanmean(lon))
        },
        zoom=11,
        height=550,
        map_style="open-street-map",
        title="Kochi Land Surface Temperature"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=45, b=0)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Map displays satellite-derived land surface temperature. "
        "Risk thresholds are based on local percentiles from the project dataset."
    )

    # --------------------------------------------------------
    # TOP HOTSPOTS
    # --------------------------------------------------------

    st.subheader("🔥 Highest-priority thermal hotspots")

    hotspots = get_hotspot_pixels(
        lst,
        ndvi_raster,
        ndbi_raster,
        ndwi_raster,
        lon,
        lat,
        percentile=95,
        max_points=10
    )

    if not hotspots.empty:

        hotspot_display = hotspots.copy()

        hotspot_display["Risk"] = hotspot_display["LST_C"].apply(
            lambda x: classify_risk(x, p90, p95)
        )

        hotspot_display["Priority Score"] = hotspot_display.apply(
            lambda row: priority_score(
                row["Risk"],
                row["NDVI"],
                row["NDBI"],
                row["NDWI"],
                row["LST_C"],
                p90,
                p95
            ),
            axis=1
        )

        hotspot_display = hotspot_display[
            [
                "Latitude",
                "Longitude",
                "LST_C",
                "Risk",
                "Priority Score",
                "NDVI",
                "NDBI",
                "NDWI"
            ]
        ]

        hotspot_display.columns = [
            "Latitude",
            "Longitude",
            "LST (°C)",
            "Risk",
            "Priority",
            "NDVI",
            "NDBI",
            "NDWI"
        ]

        st.dataframe(
            hotspot_display,
            use_container_width=True,
            hide_index=True
        )

        csv = hotspot_display.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Priority Hotspots",
            csv,
            "kochi_priority_hotspots.csv",
            "text/csv"
        )

    # --------------------------------------------------------
    # MODEL SIGNAL
    # --------------------------------------------------------

    st.subheader("🤖 What is driving the model?")

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }
    )

    importance["Importance %"] = (
        importance["Importance"] * 100
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig_imp = px.bar(
        importance,
        x="Importance %",
        y="Feature",
        orientation="h",
        text="Importance %"
    )

    fig_imp.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_imp.update_layout(
        height=350,
        margin=dict(l=0, r=20, t=20, b=20)
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )

    st.caption(
        "Feature importance indicates predictive contribution within "
        "the trained Random Forest; it should not be interpreted as causal impact."
    )


# ============================================================
# PAGE 2 — HEAT MAP
# ============================================================

elif page == "🌡️ Heat Map":

    st.title("🌡️ Environmental Heat Map")

    st.markdown(
        """
        Explore the satellite-derived environmental layers used by
        HeatShield AI.
        """
    )

    layer = st.selectbox(
        "Select environmental layer",
        [
            "Land Surface Temperature (LST)",
            "Vegetation (NDVI)",
            "Built-up Signal (NDBI)",
            "Water Signal (NDWI)"
        ]
    )

    if layer == "Land Surface Temperature (LST)":
        raster_layer = lst
        label = "LST (°C)"

    elif layer == "Vegetation (NDVI)":
        raster_layer = ndvi_raster
        label = "NDVI"

    elif layer == "Built-up Signal (NDBI)":
        raster_layer = ndbi_raster
        label = "NDBI"

    else:
        raster_layer = ndwi_raster
        label = "NDWI"

    step = max(
        1,
        int(max(raster_layer.shape) / 200)
    )

    values = raster_layer[::step, ::step]

    map_lat = lat[::step]
    map_lon = lon[::step]

    zmin = np.nanpercentile(values, 2)
    zmax = np.nanpercentile(values, 98)

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=map_lon,
            y=map_lat,
            colorscale="Turbo",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=label)
        )
    )

    fig.update_layout(
        title=layer,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=650,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The displayed environmental layers are derived from the exported "
        "satellite raster used by the prototype."
    )


# ============================================================
# PAGE 3 — HOTSPOT INTELLIGENCE
# ============================================================

elif page == "🔥 Hotspot Intelligence":

    st.title("🔥 Hotspot Intelligence")

    st.markdown(
        """
        ### Move from a temperature value to an interpretable AI assessment.

        This page combines observed satellite temperature with the
        environmental conditions surrounding each hotspot.
        """
    )

    hotspots = get_hotspot_pixels(
        lst,
        ndvi_raster,
        ndbi_raster,
        ndwi_raster,
        lon,
        lat,
        percentile=95,
        max_points=50
    )

    if hotspots.empty:

        st.warning("No hotspot pixels were detected.")

        st.stop()

    hotspot_options = []

    for i, row in hotspots.iterrows():

        hotspot_options.append(
            (
                i,
                f"Hotspot #{len(hotspot_options) + 1} — "
                f"{row['LST_C']:.2f}°C"
            )
        )

    selected_label = st.selectbox(
        "Select hotspot",
        [label for _, label in hotspot_options]
    )

    selected_index = [
        idx
        for idx, label in hotspot_options
        if label == selected_label
    ][0]

    hotspot = hotspots.loc[selected_index]

    observed_lst = float(hotspot["LST_C"])
    base_ndvi = float(hotspot["NDVI"])
    base_ndbi = float(hotspot["NDBI"])
    base_ndwi = float(hotspot["NDWI"])

    ai_lst = predict_pixel(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    risk = classify_risk(
        observed_lst,
        p90,
        p95
    )

    anomaly = observed_lst - ai_lst

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Observed LST",
            f"{observed_lst:.2f}°C",
            "Satellite observation"
        )

    with c2:
        metric_card(
            "AI Estimated LST",
            f"{ai_lst:.2f}°C",
            "Model estimate from environmental indicators"
        )

    with c3:
        metric_card(
            "Risk",
            risk,
            "Local percentile classification"
        )

    with c4:
        metric_card(
            "Thermal Gap",
            f"{anomaly:+.2f}°C",
            "Observed minus AI estimate"
        )

    # --------------------------------------------------------
    # IMPORTANT MODEL NOTE
    # --------------------------------------------------------

    if abs(anomaly) > 10:

        st.markdown(
            f"""
            <div class="warning-box">
                <strong>⚠️ Large observation-model difference</strong><br><br>
                This hotspot has a difference of approximately
                <strong>{anomaly:+.2f}°C</strong> between the satellite
                observation and the model estimate.

                The AI estimate should therefore be interpreted as a
                model-based environmental signal rather than as a replacement
                for the satellite observation.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="info-box">
                <strong>ℹ️ Interpretation</strong><br><br>
                The AI estimate represents the temperature predicted from
                the three environmental indicators used by the Random Forest.
                It is intended for hotspot interpretation and scenario analysis.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader("📍 Hotspot Location")

    loc1, loc2, loc3 = st.columns(3)

    with loc1:
        metric_card(
            "Latitude",
            f"{float(hotspot['Latitude']):.5f}°"
        )

    with loc2:
        metric_card(
            "Longitude",
            f"{float(hotspot['Longitude']):.5f}°"
        )

    with loc3:
        metric_card(
            "Priority Score",
            f"{priority_score(risk, base_ndvi, base_ndbi, base_ndwi, observed_lst, p90, p95):.0f}/100",
            "Heuristic planning score"
        )

    # --------------------------------------------------------
    # ENVIRONMENTAL PROFILE
    # --------------------------------------------------------

    st.subheader("🌱 Environmental Profile")

    e1, e2, e3 = st.columns(3)

    with e1:
        metric_card(
            "NDVI",
            f"{base_ndvi:.3f}",
            "Vegetation signal"
        )

    with e2:
        metric_card(
            "NDBI",
            f"{base_ndbi:.3f}",
            "Built-up signal"
        )

    with e3:
        metric_card(
            "NDWI",
            f"{base_ndwi:.3f}",
            "Water signal"
        )

    # --------------------------------------------------------
    # AI DIAGNOSIS
    # --------------------------------------------------------

    st.subheader("🧠 AI Environmental Diagnosis")

    diagnosis = intervention_diagnosis(
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    diagnosis_df = pd.DataFrame(
        diagnosis,
        columns=[
            "Indicator",
            "Signal",
            "Interpretation"
        ]
    )

    st.dataframe(
        diagnosis_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    recommendations = []

    if base_ndvi < 0.20:
        recommendations.append(
            "Increase tree canopy, shaded streets and green corridors."
        )

    elif base_ndvi < 0.40:
        recommendations.append(
            "Strengthen existing vegetation and prioritize shade expansion."
        )

    if base_ndbi > 0.15:
        recommendations.append(
            "Evaluate cool/reflective roofs, pavements and shaded built surfaces."
        )

    elif base_ndbi > 0:
        recommendations.append(
            "Consider reflective materials and additional urban shade."
        )

    if base_ndwi < -0.15:
        recommendations.append(
            "Evaluate feasible blue-infrastructure opportunities such as "
            "water-sensitive public spaces or restored drainage/wetland systems."
        )

    if not recommendations:
        recommendations.append(
            "Use the Cooling Simulator to compare intervention scenarios."
        )

    st.subheader("🎯 Suggested Intervention Direction")

    for recommendation in recommendations:
        st.markdown(f"**• {recommendation}**")

    st.caption(
        "These recommendations are decision-support suggestions derived "
        "from environmental indicators and model scenarios. They are not "
        "guarantees of physical cooling performance."
    )


# ============================================================
# PAGE 4 — COOLING SIMULATOR
# ============================================================

elif page == "🧊 Cooling Simulator":

    st.title("🧊 Cooling Strategy Simulator")

    st.markdown(
        """
        Test how changing environmental indicators could alter the model's
        predicted land surface temperature.
        """
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        scenario_ndvi = st.slider(
            "🌳 Vegetation improvement",
            min_value=float(np.nanpercentile(df["NDVI"], 5)),
            max_value=float(np.nanpercentile(df["NDVI"], 95)),
            value=float(np.nanmedian(df["NDVI"])),
            step=0.01
        )

    with c2:

        scenario_ndbi = st.slider(
            "🏙️ Built-up signal",
            min_value=float(np.nanpercentile(df["NDBI"], 5)),
            max_value=float(np.nanpercentile(df["NDBI"], 95)),
            value=float(np.nanmedian(df["NDBI"])),
            step=0.01
        )

    with c3:

        scenario_ndwi = st.slider(
            "💧 Water signal",
            min_value=float(np.nanpercentile(df["NDWI"], 5)),
            max_value=float(np.nanpercentile(df["NDWI"], 95)),
            value=float(np.nanmedian(df["NDWI"])),
            step=0.01
        )

    predicted = predict_pixel(
        model,
        scenario_ndvi,
        scenario_ndbi,
        scenario_ndwi
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Predicted LST",
            f"{predicted:.2f}°C",
            "Random Forest scenario estimate"
        )

    with c2:
        metric_card(
            "Vegetation",
            f"{scenario_ndvi:.3f}",
            "NDVI"
        )

    with c3:
        metric_card(
            "Water",
            f"{scenario_ndwi:.3f}",
            "NDWI"
        )

    # --------------------------------------------------------
    # PRESET SCENARIOS
    # --------------------------------------------------------

    st.subheader("🧪 Compare intervention scenarios")

    baseline_ndvi = float(np.nanmedian(df["NDVI"]))
    baseline_ndbi = float(np.nanmedian(df["NDBI"]))
    baseline_ndwi = float(np.nanmedian(df["NDWI"]))

    scenario_df, baseline = build_scenarios(
        model,
        baseline_ndvi,
        baseline_ndbi,
        baseline_ndwi,
        ndvi_p75,
        ndbi_p25,
        ndwi_p75
    )

    display_df = scenario_df.copy()

    display_df["Modeled Change"] = display_df[
        "Change vs Baseline"
    ].apply(
        lambda x: f"{x:+.2f}°C"
    )

    display_df["Predicted LST"] = display_df[
        "Predicted LST"
    ].apply(
        lambda x: f"{x:.2f}°C"
    )

    st.dataframe(
        display_df[
            [
                "Intervention",
                "Description",
                "Predicted LST",
                "Modeled Change"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    chart_df = scenario_df.copy()

    chart_df["Cooling"] = (
        -chart_df["Change vs Baseline"]
    )

    fig = px.bar(
        chart_df,
        x="Intervention",
        y="Cooling",
        text="Cooling",
        title="Modeled Temperature Change Relative to Baseline"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}°C",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Modeled cooling (°C)",
        xaxis_title="Intervention",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Scenario results are model-based estimates. They represent "
        "changes in model predictions under altered feature values, "
        "not measured or guaranteed real-world cooling."
    )


# ============================================================
# PAGE 5 — AI PLANNING
# ============================================================

elif page == "🤖 AI Planning":

    st.title("🤖 AI Cooling Planner")

    st.markdown(
        """
        ### Convert heat intelligence into a planning decision

        The planner evaluates environmental conditions and compares
        intervention scenarios for the selected hotspot.
        """
    )

    hotspots = get_hotspot_pixels(
        lst,
        ndvi_raster,
        ndbi_raster,
        ndwi_raster,
        lon,
        lat,
        percentile=95,
        max_points=100
    )

    if hotspots.empty:

        st.warning("No extreme hotspots available.")

        st.stop()

    hotspot_labels = []

    for i, row in hotspots.iterrows():

        hotspot_labels.append(
            (
                i,
                f"Hotspot #{len(hotspot_labels) + 1} — "
                f"{row['LST_C']:.2f}°C"
            )
        )

    selected_label = st.selectbox(
        "Choose a priority hotspot",
        [x[1] for x in hotspot_labels]
    )

    selected_index = [
        idx
        for idx, label in hotspot_labels
        if label == selected_label
    ][0]

    hotspot = hotspots.loc[selected_index]

    base_ndvi = float(hotspot["NDVI"])
    base_ndbi = float(hotspot["NDBI"])
    base_ndwi = float(hotspot["NDWI"])
    observed_lst = float(hotspot["LST_C"])

    scenarios, baseline = build_scenarios(
        model,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        ndvi_p75,
        ndbi_p25,
        ndwi_p75
    )

    # Largest modeled reduction
    largest_reduction_idx = scenarios[
        "Change vs Baseline"
    ].idxmin()

    largest_reduction = scenarios.loc[
        largest_reduction_idx
    ]

    # --------------------------------------------------------
    # PRIORITY SUMMARY
    # --------------------------------------------------------

    risk = classify_risk(
        observed_lst,
        p90,
        p95
    )

    score = priority_score(
        risk,
        base_ndvi,
        base_ndbi,
        base_ndwi,
        observed_lst,
        p90,
        p95
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Observed LST",
            f"{observed_lst:.2f}°C"
        )

    with c2:
        metric_card(
            "Risk",
            risk
        )

    with c3:
        metric_card(
            "Priority",
            f"{score}/100"
        )

    with c4:
        metric_card(
            "Modeled Reduction",
            f"{-largest_reduction['Change vs Baseline']:.2f}°C",
            "Largest modeled reduction among tested scenarios"
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="success-box">
            <strong>🎯 AI Planning Signal</strong><br><br>

            For this hotspot, the tested scenario producing the largest
            modeled reduction was:

            <strong>{largest_reduction['Intervention']}</strong>

            with a modeled change of

            <strong>{largest_reduction['Change vs Baseline']:+.2f}°C</strong>.

            <br><br>

            This is a model-based scenario comparison, not a claim that
            the intervention will produce exactly this temperature change
            in the real environment.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SCENARIO TABLE
    # --------------------------------------------------------

    st.subheader("📊 Intervention comparison")

    planner_df = scenarios.copy()

    planner_df["Modeled Reduction"] = (
        -planner_df["Change vs Baseline"]
    )

    planner_df["Predicted LST"] = planner_df[
        "Predicted LST"
    ].round(2)

    planner_df["Modeled Reduction"] = planner_df[
        "Modeled Reduction"
    ].round(2)

    st.dataframe(
        planner_df[
            [
                "Intervention",
                "Description",
                "Predicted LST",
                "Modeled Reduction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    csv = planner_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download AI Planning Analysis",
        csv,
        "kochi_ai_cooling_plan.csv",
        "text/csv"
    )

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    st.subheader("🧠 Why this location needs attention")

    reasons = []

    if observed_lst >= p95:
        reasons.append(
            "The observed surface temperature falls within the local extreme-heat percentile."
        )

    if base_ndvi < 0.20:
        reasons.append(
            "The vegetation signal is relatively low."
        )

    if base_ndbi > 0.15:
        reasons.append(
            "The built-up signal is relatively high."
        )

    if base_ndwi < -0.15:
        reasons.append(
            "The water signal is relatively low."
        )

    if not reasons:
        reasons.append(
            "The hotspot is thermally significant even though the environmental indicators "
            "do not cross the strongest heuristic thresholds."
        )

    for reason in reasons:
        st.markdown(f"**• {reason}**")

    st.caption(
        "Priority scoring and intervention recommendations are transparent "
        "heuristics layered on top of the machine-learning model."
    )


# ============================================================
# PAGE 6 — MODEL & DATA
# ============================================================

elif page == "📊 Model & Data":

    st.title("📊 Model, Data & Validation")

    st.markdown(
        """
        ### Transparent AI

        HeatShield AI uses satellite-derived environmental indicators
        to estimate land surface temperature and explore cooling scenarios.
        """
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("🤖 Random Forest Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "R²",
            f"{float(metadata.get('r2', 0)):.4f}",
            "Held-out test set"
        )

    with c2:
        metric_card(
            "RMSE",
            f"{float(metadata.get('rmse', 0)):.4f}°C",
            "Held-out test set"
        )

    with c3:
        metric_card(
            "MAE",
            f"{float(metadata.get('mae', 0)):.4f}°C",
            "Held-out test set"
        )

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.subheader("⚙️ Model Configuration")

    model_info = pd.DataFrame(
        {
            "Parameter": [
                "Algorithm",
                "Estimators",
                "Maximum depth",
                "Training samples",
                "Test samples",
                "Features",
                "Target"
            ],
            "Value": [
                metadata.get("model", "Random Forest"),
                metadata.get("n_estimators", "200"),
                metadata.get("max_depth", "15"),
                metadata.get("training_samples", "—"),
                metadata.get("test_samples", "—"),
                ", ".join(FEATURES),
                metadata.get("target", "LST")
            ]
        }
    )

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }
    )

    importance["Importance %"] = (
        importance["Importance"] * 100
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        importance,
        x="Feature",
        y="Importance %",
        text="Importance %"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Feature importance describes predictive importance within the "
        "Random Forest and does not establish causation."
    )

    # --------------------------------------------------------
    # DATA DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🌡️ LST Distribution")

    fig = px.histogram(
        df,
        x="LST",
        nbins=50,
        title="Training Dataset Land Surface Temperature"
    )

    fig.add_vline(
        x=p90,
        line_dash="dash",
        annotation_text=f"P90 = {p90:.2f}°C"
    )

    fig.add_vline(
        x=p95,
        line_dash="dash",
        annotation_text=f"P95 = {p95:.2f}°C"
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RISK THRESHOLDS
    # --------------------------------------------------------

    st.subheader("🚦 Heat Risk Thresholds")

    threshold_df = pd.DataFrame(
        {
            "Risk": [
                "Low",
                "Moderate",
                "High",
                "Extreme"
            ],
            "Definition": [
                f"Below {(p90 + p95) / 2:.2f}°C",
                f"{(p90 + p95) / 2:.2f}°C to < {p90:.2f}°C",
                f"{p90:.2f}°C to < {p95:.2f}°C",
                f">= {p95:.2f}°C"
            ]
        }
    )

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "These thresholds are percentile-based thresholds derived from "
        "the project dataset. They are not universal public-health heat "
        "warning thresholds."
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.subheader("⚠️ Important Model Limitations")

    st.markdown(
        """
        **1. Surface temperature is not the same as air temperature.**

        The system works with satellite-derived land surface temperature.

        **2. Feature importance is not causal evidence.**

        A high Random Forest feature importance does not mean that changing
        that feature alone will produce the same percentage of temperature
        change.

        **3. Scenario outputs are model simulations.**

        Cooling estimates are changes in model prediction after modifying
        environmental indicators. They are not guaranteed physical cooling.

        **4. The prototype is designed for decision support.**

        Final infrastructure decisions should incorporate local engineering,
        cost, land availability, drainage, public-health and community data.

        **5. Spatial and temporal coverage matter.**

        Satellite observations represent particular acquisition conditions
        and should be combined with additional meteorological and urban data
        for operational deployment.
        """
    )

    # --------------------------------------------------------
    # DATA DOWNLOAD
    # --------------------------------------------------------

    st.subheader("📥 Project Data")

    data_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Training Dataset",
        data_csv,
        "kochi_heat_data.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#777;font-size:12px;">
        🌡️ <strong>HeatShield AI</strong> —
        AI-Powered Urban Heat Intelligence & Cooling Optimizer<br>
        Kochi Urban Heat Study • Satellite + Machine Learning + Scenario Planning
    </div>
    """,
    unsafe_allow_html=True
)