import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tifffile
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# HEATSHIELD AI v2
# KOCHI URBAN HEAT INTELLIGENCE & COOLING OPTIMIZER
# ============================================================

st.set_page_config(
    page_title="HeatShield AI | Kochi",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #243b53);
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-size: 2.6rem;
    margin-bottom: 0.2rem;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.88;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}

.metric-card {
    background: white;
    padding: 1rem 1.1rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    min-height: 110px;
}

.metric-label {
    font-size: 0.82rem;
    color: #6b7280;
    font-weight: 600;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 750;
    margin-top: 0.35rem;
}

.metric-note {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.25rem;
}

.insight {
    background: #eef6ff;
    border-left: 5px solid #2563eb;
    padding: 1rem 1.2rem;
    border-radius: 8px;
}

.warning-box {
    background: #fff8e6;
    border-left: 5px solid #f59e0b;
    padding: 1rem 1.2rem;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("kochi_heat_data.csv")

    df = df[
        ["LST", "NDVI", "NDBI", "NDWI"]
    ].dropna().copy()

    return df


@st.cache_resource
def load_model():
    return joblib.load("kochi_heat_model.pkl")


@st.cache_data
def load_metadata():
    with open("model_metadata.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_raster():

    with tifffile.TiffFile(
        "Kochi_HeatShield_Raster.tif"
    ) as tif:

        page = tif.pages[0]

        raster = page.asarray()

        # Read GeoTIFF geographic referencing
        pixel_scale = page.tags[
            "ModelPixelScaleTag"
        ].value

        tiepoint = page.tags[
            "ModelTiepointTag"
        ].value

    # --------------------------------------------------------
    # GEOGRAPHIC TRANSFORM
    # --------------------------------------------------------

    pixel_width = pixel_scale[0]
    pixel_height = pixel_scale[1]

    origin_lon = tiepoint[3]
    origin_lat = tiepoint[4]

    height = raster.shape[0]
    width = raster.shape[1]

    # Pixel-center coordinates
    lon = (
        origin_lon
        + np.arange(width) * pixel_width
    )

    lat = (
        origin_lat
        - np.arange(height) * pixel_height
    )

    return raster, lon, lat


df = load_data()
model = load_model()
metadata = load_metadata()
raster, lon, lat = load_raster()

# ============================================================
# RASTER INFORMATION
# ============================================================

# GeoTIFF should contain:
# Band 1 = LST
# Band 2 = NDVI
# Band 3 = NDBI
# Band 4 = NDWI

if raster.ndim == 3:

    if raster.shape[0] == 4:
        lst_raster = raster[0]
        ndvi_raster = raster[1]
        ndbi_raster = raster[2]
        ndwi_raster = raster[3]

    elif raster.shape[2] == 4:
        lst_raster = raster[:, :, 0]
        ndvi_raster = raster[:, :, 1]
        ndbi_raster = raster[:, :, 2]
        ndwi_raster = raster[:, :, 3]

    else:
        lst_raster = raster[0]
        ndvi_raster = raster[1]
        ndbi_raster = raster[2]
        ndwi_raster = raster[3]

else:
    lst_raster = raster


# ============================================================
# MODEL / STATISTICS
# ============================================================

FEATURES = [
    "NDVI",
    "NDBI",
    "NDWI"
]

median_lst = df["LST"].median()
mean_lst = df["LST"].mean()
p90 = df["LST"].quantile(0.90)
p95 = df["LST"].quantile(0.95)

ndvi_p75 = df["NDVI"].quantile(0.75)
ndbi_p25 = df["NDBI"].quantile(0.25)
ndwi_p75 = df["NDWI"].quantile(0.75)


def predict_temperature(
    ndvi,
    ndbi,
    ndwi
):

    X = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "NDWI": [ndwi]
    })

    return float(
        model.predict(X)[0]
    )


def heat_category(temp):

    if temp >= p95:
        return "Extreme"

    elif temp >= p90:
        return "High"

    elif temp >= median_lst:
        return "Moderate"

    else:
        return "Low"


df["Heat_Risk"] = df["LST"].apply(
    heat_category
)


# ============================================================
# SIDEBAR
# ============================================================



st.sidebar.markdown(
    """
    # 🌡️ HeatShield AI

    **Kochi Urban Heat Intelligence**
    """
)

st.sidebar.divider()

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

st.sidebar.divider()

st.sidebar.caption(
    "Satellite-derived environmental indicators"
)

st.sidebar.caption(
    "Random Forest heat prediction"
)

st.sidebar.caption(
    "Scenario-based cooling optimization"
)


# ============================================================
# HERO
# ============================================================

def show_hero():

    st.markdown(
        """
        <div class="hero">

        <h1>🌡️ HeatShield AI</h1>

        <p>
        AI-Powered Urban Heat Intelligence & Cooling Optimizer
        </p>

        <p>
        <b>Study Area:</b> Kochi, Kerala
        &nbsp; • &nbsp;
        Satellite + Machine Learning
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    show_hero()

    st.markdown(
        '<div class="section-title">Kochi Heat Intelligence</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    extreme_count = int(
        (df["LST"] >= p95).sum()
    )

    high_count = int(
        (df["LST"] >= p90).sum()
    )

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">PIXELS ANALYZED</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-note">Sampled Kochi locations</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">MEAN LAND SURFACE TEMP.</div>
            <div class="metric-value">{mean_lst:.2f}°C</div>
            <div class="metric-note">Observed satellite LST</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">P90 HEAT THRESHOLD</div>
            <div class="metric-value">{p90:.2f}°C</div>
            <div class="metric-note">High-risk threshold</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-label">EXTREME-RISK PIXELS</div>
            <div class="metric-value">{extreme_count:,}</div>
            <div class="metric-note">At or above P95</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

  
   # --------------------------------------------------------
# GEOGRAPHICALLY REFERENCED HEAT MAP
# --------------------------------------------------------

st.markdown(
    '<div class="section-title">🔥 Kochi Urban Heat Surface</div>',
    unsafe_allow_html=True
)

display_raster = lst_raster.copy()

# Downsample for browser performance
max_dimension = 500

step = max(
    1,
    int(
        max(display_raster.shape) / max_dimension
    )
)

display_raster = display_raster[
    ::step,
    ::step
]

display_lon = lon[
    ::step
]

display_lat = lat[
    ::step
]

# Replace invalid values
display_raster = np.where(
    np.isfinite(display_raster),
    display_raster,
    np.nan
)

# Geographic heat map
fig = go.Figure(
    data=go.Heatmap(
        z=display_raster,
        x=display_lon,
        y=display_lat,
        colorscale="Turbo",
        colorbar=dict(
            title="LST °C"
        ),
        hovertemplate=(
            "Longitude: %{x:.5f}°E"
            "<br>Latitude: %{y:.5f}°N"
            "<br>LST: %{z:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    height=600,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    xaxis=dict(
        title="Longitude (°E)",
        tickformat=".2f"
    ),
    yaxis=dict(
        title="Latitude (°N)",
        tickformat=".2f"
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.caption(
    "Georeferenced Landsat-derived LST raster. "
    "Hover over the map to inspect longitude, latitude, "
    "and land-surface temperature."
)

    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="section-title">🔥 Heat Risk Distribution</div>',
            unsafe_allow_html=True
        )

        risk_counts = (
            df["Heat_Risk"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Moderate",
                    "High",
                    "Extreme"
                ],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk",
            "Pixels"
        ]

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Pixels",
            text="Pixels"
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with c2:

        st.markdown(
            '<div class="section-title">🧠 Model Signals</div>',
            unsafe_allow_html=True
        )

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            text_auto=".1%"
        )

        fig.update_layout(
            height=350,
            xaxis_title="Predictive importance"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.warning(
        "⚠️ Risk categories are percentile-based within this study dataset. "
        "They are not a universal weather or health standard."
    )


# ============================================================
# HEAT MAP PAGE
# ============================================================

elif page == "🌡️ Heat Map":

    show_hero()

    st.header("🌡️ Spatial Heat Map")

    st.write(
        "Explore the satellite-derived land surface temperature "
        "surface exported for the Kochi study area."
    )

    # --------------------------------------------------------
    # LAYER SELECTOR
    # --------------------------------------------------------

    layer = st.selectbox(
        "Select environmental layer",
        [
            "LST — Land Surface Temperature",
            "NDVI — Vegetation",
            "NDBI — Built Environment",
            "NDWI — Water"
        ]
    )

    if layer.startswith("LST"):

        selected_raster = lst_raster
        color_scale = "Turbo"
        label = "LST"

    elif layer.startswith("NDVI"):

        selected_raster = ndvi_raster
        color_scale = "Greens"
        label = "NDVI"

    elif layer.startswith("NDBI"):

        selected_raster = ndbi_raster
        color_scale = "Oranges"
        label = "NDBI"

    else:

        selected_raster = ndwi_raster
        color_scale = "Blues"
        label = "NDWI"

    selected_raster = np.where(
        np.isfinite(selected_raster),
        selected_raster,
        np.nan
    )

    step = max(
        1,
        int(
            max(
                selected_raster.shape
            ) / 700
        )
    )

    display_layer = selected_raster[
        ::step,
        ::step
    ]

   fig = go.Figure(
    data=go.Heatmap(
        z=display_layer,
        x=lon[::step],
        y=lat[::step],
        colorscale=color_scale,
        colorbar=dict(
            title=label
        ),
        hovertemplate=(
            "Longitude: %{x:.5f}°E"
            "<br>Latitude: %{y:.5f}°N"
            f"<br>{label}: %{{z:.3f}}"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    height=700,
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    xaxis=dict(
        title="Longitude (°E)",
        tickformat=".2f"
    ),
    yaxis=dict(
        title="Latitude (°N)",
        tickformat=".2f"
    )
)
    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # LAYER STATISTICS
    # --------------------------------------------------------

    valid = selected_raster[
        np.isfinite(selected_raster)
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum",
            f"{np.nanmin(valid):.3f}"
        )

    with c2:
        st.metric(
            "Median",
            f"{np.nanmedian(valid):.3f}"
        )

    with c3:
        st.metric(
            "Mean",
            f"{np.nanmean(valid):.3f}"
        )

    with c4:
        st.metric(
            "Maximum",
            f"{np.nanmax(valid):.3f}"
        )

    st.info(
        "The GeoTIFF is visualized directly from the exported raster. "
        "Because Rasterio is unavailable in this Windows environment, "
        "this prototype displays the raster in image coordinates rather "
        "than adding geographic axes."
    )


# ============================================================
# HOTSPOT INTELLIGENCE
# ============================================================

elif page == "🔥 Hotspot Intelligence":

    show_hero()

    st.header("🔥 Hotspot Intelligence")

    st.write(
        "Move from a temperature value to an interpretable AI assessment."
    )

    hotspots = (
        df.sort_values(
            "LST",
            ascending=False
        )
        .head(100)
        .reset_index(drop=True)
    )

    selected = st.selectbox(
        "Select hotspot",
        range(len(hotspots)),
        format_func=lambda i:
        f"Hotspot #{i + 1} — "
        f"{hotspots.iloc[i]['LST']:.2f}°C"
    )

    hotspot = hotspots.iloc[selected]

    observed = float(
        hotspot["LST"]
    )

    ndvi = float(
        hotspot["NDVI"]
    )

    ndbi = float(
        hotspot["NDBI"]
    )

    ndwi = float(
        hotspot["NDWI"]
    )

    prediction = predict_temperature(
        ndvi,
        ndbi,
        ndwi
    )

    risk = heat_category(
        observed
    )

    # --------------------------------------------------------
    # TOP CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Observed LST",
            f"{observed:.2f}°C"
        )

    with c2:
        st.metric(
            "AI Estimated LST",
            f"{prediction:.2f}°C"
        )

    with c3:
        st.metric(
            "Risk",
            risk
        )

    with c4:
        st.metric(
            "Observed − AI",
            f"{observed - prediction:+.2f}°C"
        )

    st.divider()

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    st.subheader("🌱 Environmental Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "NDVI",
            f"{ndvi:.3f}"
        )

    with c2:
        st.metric(
            "NDBI",
            f"{ndbi:.3f}"
        )

    with c3:
        st.metric(
            "NDWI",
            f"{ndwi:.3f}"
        )

    # --------------------------------------------------------
    # MODEL IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Predictive Signals"
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".1%"
    )

    fig.update_layout(
        height=300
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.markdown(
        """
        <div class="insight">

        <b>Interpretation:</b>

        The model uses NDVI, NDBI and NDWI together to estimate LST.
        Feature importance describes predictive contribution within
        this Random Forest; it does not prove that changing a feature
        will physically cause the displayed temperature change.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TOP HOTSPOTS
    # --------------------------------------------------------

    st.subheader(
        "🔥 Top 20 Observed Hotspots"
    )

    table = hotspots.head(20).copy()

    table.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(table) + 1
        )
    )

    st.dataframe(
        table.style.format({
            "LST": "{:.2f}",
            "NDVI": "{:.3f}",
            "NDBI": "{:.3f}",
            "NDWI": "{:.3f}"
        }),
        width="stretch"
    )


# ============================================================
# COOLING SIMULATOR
# ============================================================

elif page == "🧊 Cooling Simulator":

    show_hero()

    st.header("🧊 Cooling Strategy Simulator")

    st.write(
        "Test intervention scenarios by moving environmental indicators "
        "toward values observed elsewhere in the study dataset."
    )

    st.markdown(
        """
        <div class="warning-box">

        <b>Scientific note:</b>

        These are machine-learning scenario responses.
        They are not direct measurements of physical cooling from
        planting trees, adding water infrastructure, changing roofs,
        or modifying buildings.

        </div>
        """,
        unsafe_allow_html=True
    )

    hotspots = (
        df.sort_values(
            "LST",
            ascending=False
        )
        .head(100)
        .reset_index(drop=True)
    )

    selected = st.selectbox(
        "Select hotspot",
        range(len(hotspots)),
        format_func=lambda i:
        f"Hotspot #{i + 1} — "
        f"{hotspots.iloc[i]['LST']:.2f}°C"
    )

    base = hotspots.iloc[selected]

    base_ndvi = float(
        base["NDVI"]
    )

    base_ndbi = float(
        base["NDBI"]
    )

    base_ndwi = float(
        base["NDWI"]
    )

    baseline = predict_temperature(
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    st.divider()

    # --------------------------------------------------------
    # SLIDERS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        green = st.slider(
            "🌳 Green Infrastructure",
            0,
            100,
            0,
            step=10
        )

    with c2:

        blue = st.slider(
            "💧 Blue Infrastructure",
            0,
            100,
            0,
            step=10
        )

    with c3:

        built = st.slider(
            "🏢 Built Environment",
            0,
            100,
            0,
            step=10
        )

    # --------------------------------------------------------
    # SCENARIO
    # --------------------------------------------------------

    if base_ndvi < ndvi_p75:

        scenario_ndvi = (
            base_ndvi +
            (green / 100) *
            (ndvi_p75 - base_ndvi)
        )

    else:

        scenario_ndvi = base_ndvi

    scenario_ndwi = (
        base_ndwi +
        (blue / 100) *
        (ndwi_p75 - base_ndwi)
    )

    if base_ndbi > ndbi_p25:

        scenario_ndbi = (
            base_ndbi -
            (built / 100) *
            (base_ndbi - ndbi_p25)
        )

    else:

        scenario_ndbi = base_ndbi

    scenario = predict_temperature(
        scenario_ndvi,
        scenario_ndbi,
        scenario_ndwi
    )

    change = scenario - baseline

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Baseline AI Prediction",
            f"{baseline:.2f}°C"
        )

    with c2:

        st.metric(
            "Scenario AI Prediction",
            f"{scenario:.2f}°C"
        )

    with c3:

        st.metric(
            "Scenario Change",
            f"{change:+.2f}°C"
        )

    # --------------------------------------------------------
    # FEATURE TRANSFORMATION
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Indicator": [
            "NDVI",
            "NDBI",
            "NDWI"
        ],
        "Baseline": [
            base_ndvi,
            base_ndbi,
            base_ndwi
        ],
        "Scenario": [
            scenario_ndvi,
            scenario_ndbi,
            scenario_ndwi
        ]
    })

    st.subheader(
        "📊 Environmental Scenario"
    )

    st.dataframe(
        comparison.style.format({
            "Baseline": "{:.3f}",
            "Scenario": "{:.3f}"
        }),
        width="stretch"
    )

    # --------------------------------------------------------
    # SCENARIO GAUGE
    # --------------------------------------------------------

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=scenario,
            title={
                "text": "AI Scenario LST"
            },
            gauge={
                "axis": {
                    "range": [
                        float(df["LST"].min()),
                        float(df["LST"].quantile(0.99))
                    ]
                }
            }
        )
    )

    fig.update_layout(
        height=350
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    if change < -0.1:

        st.success(
            f"Under the selected scenario, the model predicts "
            f"{abs(change):.2f}°C lower LST than the baseline."
        )

    elif change > 0.1:

        st.warning(
            f"Under the selected scenario, the model predicts "
            f"{change:.2f}°C higher LST than the baseline."
        )

    else:

        st.info(
            "The selected intervention settings produce little "
            "change in the model prediction."
        )


# ============================================================
# AI PLANNING
# ============================================================

elif page == "🤖 AI Planning":

    show_hero()

    st.header("🤖 AI Planning Assistant")

    st.write(
        "Compare intervention scenarios and translate the model response "
        "into a planning-oriented insight."
    )

    hotspot = (
        df.sort_values(
            "LST",
            ascending=False
        )
        .iloc[0]
    )

    base_ndvi = float(
        hotspot["NDVI"]
    )

    base_ndbi = float(
        hotspot["NDBI"]
    )

    base_ndwi = float(
        hotspot["NDWI"]
    )

    observed = float(
        hotspot["LST"]
    )

    baseline = predict_temperature(
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    # Green
    green_ndvi = (
        ndvi_p75
        if base_ndvi < ndvi_p75
        else base_ndvi
    )

    green_prediction = predict_temperature(
        green_ndvi,
        base_ndbi,
        base_ndwi
    )

    # Blue
    blue_prediction = predict_temperature(
        base_ndvi,
        base_ndbi,
        ndwi_p75
    )

    # Built
    built_ndbi = (
        ndbi_p25
        if base_ndbi > ndbi_p25
        else base_ndbi
    )

    built_prediction = predict_temperature(
        base_ndvi,
        built_ndbi,
        base_ndwi
    )

    scenarios = pd.DataFrame({
        "Strategy": [
            "Green Infrastructure",
            "Blue Infrastructure",
            "Built Environment"
        ],
        "AI Prediction": [
            green_prediction,
            blue_prediction,
            built_prediction
        ]
    })

    scenarios["Change vs Baseline"] = (
        scenarios["AI Prediction"]
        - baseline
    )

    # --------------------------------------------------------
    # HOTSPOT
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Priority Hotspot LST",
            f"{observed:.2f}°C"
        )

    with c2:

        st.metric(
            "AI Baseline",
            f"{baseline:.2f}°C"
        )

    with c3:

        st.metric(
            "Risk",
            heat_category(observed)
        )

    st.divider()

    # --------------------------------------------------------
    # SCENARIOS
    # --------------------------------------------------------

    st.subheader(
        "🧊 Full-Intensity Scenario Comparison"
    )

    st.dataframe(
        scenarios.style.format({
            "AI Prediction": "{:.2f}",
            "Change vs Baseline": "{:+.2f}"
        }),
        width="stretch"
    )

    fig = px.bar(
        scenarios,
        x="Strategy",
        y="Change vs Baseline",
        text_auto=".2f"
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    fig.update_layout(
        height=400,
        title="Model Scenario Response"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # PLANNING SIGNAL
    # --------------------------------------------------------

    best_idx = scenarios[
        "Change vs Baseline"
    ].idxmin()

    strategy = scenarios.loc[
        best_idx,
        "Strategy"
    ]

    scenario_change = scenarios.loc[
        best_idx,
        "Change vs Baseline"
    ]

    st.markdown(
        f"""
        <div class="insight">

        <h3>💡 Planning Signal</h3>

        Under the tested model scenarios, <b>{strategy}</b>
        produces the largest decrease in predicted LST for the
        selected priority hotspot.

        <br><br>

        Model scenario change:
        <b>{scenario_change:+.2f}°C</b>

        <br><br>

        This signal can be used to prioritize locations for further
        engineering, urban-design and field-validation analysis.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.warning(
        "This is a model-based prioritization signal, not a guarantee "
        "of physical cooling. Real intervention planning should consider "
        "land availability, cost, water demand, maintenance, equity, "
        "and field measurements."
    )


# ============================================================
# MODEL & DATA
# ============================================================

elif page == "📊 Model & Data":

    show_hero()

    st.header("📊 Model & Data")

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Machine Learning Performance"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R²",
            f"{metadata['r2']:.4f}"
        )

    with c2:

        st.metric(
            "RMSE",
            f"{metadata['rmse']:.4f}°C"
        )

    with c3:

        st.metric(
            "MAE",
            f"{metadata['mae']:.4f}°C"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    st.subheader(
        "🌳 Random Forest Configuration"
    )

    model_info = pd.DataFrame({
        "Parameter": [
            "Model",
            "Target",
            "Predictors",
            "Trees",
            "Maximum Depth",
            "Training Samples",
            "Test Samples"
        ],
        "Value": [
            metadata["model"],
            metadata["target"],
            ", ".join(
                metadata["features"]
            ),
            metadata["n_estimators"],
            metadata["max_depth"],
            metadata["training_samples"],
            metadata["test_samples"]
        ]
    })

    st.dataframe(
        model_info,
        hide_index=True,
        width="stretch"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🧠 Predictive Feature Importance"
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance["Percentage"] = (
        importance["Importance"] * 100
    )

    st.dataframe(
        importance.style.format({
            "Importance": "{:.4f}",
            "Percentage": "{:.2f}%"
        }),
        hide_index=True,
        width="stretch"
    )

    # --------------------------------------------------------
    # DATA DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📈 Observed Feature Distributions"
    )

    selected_feature = st.selectbox(
        "Select feature",
        [
            "LST",
            "NDVI",
            "NDBI",
            "NDWI"
        ]
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        nbins=40,
        title=f"{selected_feature} Distribution"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.info(
        "The dataset contains 9,997 sampled pixels from the exported "
        "Kochi satellite analysis. Heat-risk categories are calculated "
        "from percentiles within this dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HeatShield AI • Kochi • Urban Heat Intelligence • "
    "Satellite-derived indicators + Random Forest"
) 