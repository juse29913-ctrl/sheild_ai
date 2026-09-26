import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import plotly.express as px

# ============================================================
# HEATSHIELD AI — KOCHI URBAN HEAT INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="HeatShield AI | Kochi",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("kochi_heat_data.csv")
    return df


@st.cache_resource
def load_model():
    return joblib.load("kochi_heat_model.pkl")


@st.cache_data
def load_metadata():
    with open("model_metadata.json", "r") as f:
        return json.load(f)


df = load_data()
model = load_model()
metadata = load_metadata()

# ============================================================
# PREPARE DATA
# ============================================================

FEATURES = ["NDVI", "NDBI", "NDWI"]

df = df[["LST", "NDVI", "NDBI", "NDWI"]].dropna().copy()

median_lst = df["LST"].median()
p90 = df["LST"].quantile(0.90)
p95 = df["LST"].quantile(0.95)

# Feature ranges used by the simulator
ndvi_min = df["NDVI"].min()
ndvi_max = df["NDVI"].max()
ndvi_p75 = df["NDVI"].quantile(0.75)

ndbi_min = df["NDBI"].min()
ndbi_max = df["NDBI"].max()
ndbi_p25 = df["NDBI"].quantile(0.25)

ndwi_min = df["NDWI"].min()
ndwi_max = df["NDWI"].max()
ndwi_p75 = df["NDWI"].quantile(0.75)


def predict_temperature(ndvi, ndbi, ndwi):
    X = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "NDWI": [ndwi]
    })

    return float(model.predict(X)[0])


def heat_category(temp):
    if temp >= p95:
        return "Extreme"
    elif temp >= p90:
        return "High"
    elif temp >= median_lst:
        return "Moderate"
    else:
        return "Low"


df["Heat_Risk"] = df["LST"].apply(heat_category)

# ============================================================
# HEADER
# ============================================================

st.title("🌡️ HeatShield AI")
st.subheader("AI-Powered Urban Heat Intelligence & Cooling Optimizer")

st.markdown(
    """
    **Study Area: Kochi, Kerala**

    HeatShield AI combines satellite-derived environmental indicators
    with machine learning to identify heat-risk areas, explain hotspot
    characteristics, and simulate potential cooling strategies.
    """
)

st.divider()

# ============================================================
# IMPORTANT SCIENTIFIC DISCLAIMER
# ============================================================

st.warning(
    "⚠️ **Model interpretation:** Cooling results are scenario predictions "
    "from the trained machine-learning model after modifying environmental "
    "indicators. They are NOT direct measurements of intervention-induced "
    "cooling. Feature importance indicates predictive association, not causation."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ HeatShield AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏙️ Overview",
        "🔥 Hotspot Explorer",
        "🧊 Cooling Simulator",
        "🤖 AI Recommendation",
        "📊 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Kochi Urban Heat Intelligence")
st.sidebar.caption("Satellite + Machine Learning")

# ============================================================
# OVERVIEW
# ============================================================

if page == "🏙️ Overview":

    st.header("🏙️ Kochi Heat Intelligence Overview")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pixels Analyzed",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Mean LST",
            f"{df['LST'].mean():.2f} °C"
        )

    with col3:
        st.metric(
            "90th Percentile",
            f"{p90:.2f} °C"
        )

    with col4:
        extreme_count = int((df["LST"] >= p95).sum())

        st.metric(
            "Extreme Heat Pixels",
            f"{extreme_count:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # HEAT DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.subheader("🌡️ Land Surface Temperature Distribution")

        fig = px.histogram(
            df,
            x="LST",
            nbins=40,
            labels={"LST": "Land Surface Temperature (°C)"},
            title="Observed LST Distribution"
        )

        fig.add_vline(
            x=p90,
            line_dash="dash",
            annotation_text=f"P90 = {p90:.2f}°C"
        )

        fig.add_vline(
            x=p95,
            line_dash="dot",
            annotation_text=f"P95 = {p95:.2f}°C"
        )

        fig.update_layout(
            height=450,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("🔥 Heat Risk Breakdown")

        risk_counts = (
            df["Heat_Risk"]
            .value_counts()
            .reindex(
                ["Low", "Moderate", "High", "Extreme"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = ["Risk", "Pixels"]

        fig = px.pie(
            risk_counts,
            names="Risk",
            values="Pixels",
            hole=0.45
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🧠 What does the AI use to estimate temperature?")

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
        xaxis_title="Predictive Feature Importance",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Feature importance describes how strongly each variable contributes "
        "to the Random Forest's predictions in this dataset. It should not "
        "be interpreted as proof of a causal relationship."
    )

# ============================================================
# HOTSPOT EXPLORER
# ============================================================

elif page == "🔥 Hotspot Explorer":

    st.header("🔥 Hotspot Explorer")

    st.write(
        "Explore pixels with elevated observed land surface temperature "
        "and inspect the environmental indicators associated with them."
    )

    # --------------------------------------------------------
    # HOTSPOT SELECTION
    # --------------------------------------------------------

    hotspots = df.sort_values(
        "LST",
        ascending=False
    ).head(100).reset_index(drop=True)

    hotspot_number = st.slider(
        "Select hotspot",
        min_value=1,
        max_value=len(hotspots),
        value=1
    )

    hotspot = hotspots.iloc[hotspot_number - 1]

    observed_lst = float(hotspot["LST"])
    ndvi = float(hotspot["NDVI"])
    ndbi = float(hotspot["NDBI"])
    ndwi = float(hotspot["NDWI"])

    predicted_lst = predict_temperature(
        ndvi,
        ndbi,
        ndwi
    )

    risk = heat_category(observed_lst)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Observed LST",
            f"{observed_lst:.2f} °C"
        )

    with c2:
        st.metric(
            "AI Estimated LST",
            f"{predicted_lst:.2f} °C"
        )

    with c3:
        st.metric(
            "Heat Risk",
            risk
        )

    with c4:
        st.metric(
            "Model Difference",
            f"{observed_lst - predicted_lst:+.2f} °C"
        )

    st.divider()

    # --------------------------------------------------------
    # ENVIRONMENTAL INDICATORS
    # --------------------------------------------------------

    st.subheader("🌱 Environmental Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "NDVI — Vegetation",
            f"{ndvi:.3f}"
        )

    with c2:
        st.metric(
            "NDBI — Built Environment",
            f"{ndbi:.3f}"
        )

    with c3:
        st.metric(
            "NDWI — Water",
            f"{ndwi:.3f}"
        )

    st.divider()

    # --------------------------------------------------------
    # HOTSPOT TABLE
    # --------------------------------------------------------

    st.subheader("🔥 Highest Temperature Pixels")

    display_df = hotspots.copy()

    display_df.index = np.arange(
        1,
        len(display_df) + 1
    )

    st.dataframe(
        display_df.style.format({
            "LST": "{:.2f}",
            "NDVI": "{:.3f}",
            "NDBI": "{:.3f}",
            "NDWI": "{:.3f}"
        }),
        use_container_width=True
    )

# ============================================================
# COOLING SIMULATOR
# ============================================================

elif page == "🧊 Cooling Simulator":

    st.header("🧊 AI Cooling Strategy Simulator")

    st.write(
        "Test how the trained model responds when environmental indicators "
        "are moved toward conditions observed elsewhere in the Kochi study area."
    )

    st.info(
        "The sliders represent scenario intensity. They do not represent "
        "a guaranteed physical temperature reduction."
    )

    # --------------------------------------------------------
    # HOTSPOT
    # --------------------------------------------------------

    hotspots = df.sort_values(
        "LST",
        ascending=False
    ).head(100).reset_index(drop=True)

    selected = st.selectbox(
        "Choose a hotspot",
        range(len(hotspots)),
        format_func=lambda x:
        f"Hotspot #{x+1} — {hotspots.iloc[x]['LST']:.2f} °C"
    )

    base = hotspots.iloc[selected]

    base_ndvi = float(base["NDVI"])
    base_ndbi = float(base["NDBI"])
    base_ndwi = float(base["NDWI"])

    current_prediction = predict_temperature(
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    st.divider()

    # --------------------------------------------------------
    # SLIDERS
    # --------------------------------------------------------

    green = st.slider(
        "🌳 Green Infrastructure",
        0,
        100,
        0,
        step=10,
        help="Moves NDVI toward the 75th percentile of observed values when the selected pixel is below it."
    )

    blue = st.slider(
        "💧 Blue Infrastructure",
        0,
        100,
        0,
        step=10,
        help="Moves NDWI toward the 75th percentile of observed values."
    )

    built = st.slider(
        "🏢 Built-Environment Intervention",
        0,
        100,
        0,
        step=10,
        help="Moves NDBI toward the 25th percentile when the selected pixel is above it."
    )

    # --------------------------------------------------------
    # SCENARIO FEATURES
    # --------------------------------------------------------

    # Green intervention:
    # only move toward p75 if base is below p75
    if base_ndvi < ndvi_p75:
        target_ndvi = ndvi_p75
        scenario_ndvi = (
            base_ndvi +
            (green / 100) *
            (target_ndvi - base_ndvi)
        )
    else:
        scenario_ndvi = base_ndvi

    # Blue intervention
    scenario_ndwi = (
        base_ndwi +
        (blue / 100) *
        (ndwi_p75 - base_ndwi)
    )

    # Built intervention:
    # only reduce NDBI if current value is above p25
    if base_ndbi > ndbi_p25:
        scenario_ndbi = (
            base_ndbi -
            (built / 100) *
            (base_ndbi - ndbi_p25)
        )
    else:
        scenario_ndbi = base_ndbi

    scenario_prediction = predict_temperature(
        scenario_ndvi,
        scenario_ndbi,
        scenario_ndwi
    )

    change = scenario_prediction - current_prediction

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Baseline AI Prediction",
            f"{current_prediction:.2f} °C"
        )

    with c2:
        st.metric(
            "Scenario AI Prediction",
            f"{scenario_prediction:.2f} °C"
        )

    with c3:
        st.metric(
            "Scenario Change",
            f"{change:+.2f} °C"
        )

    # --------------------------------------------------------
    # FEATURE COMPARISON
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

    st.subheader("📊 Environmental Scenario")

    st.dataframe(
        comparison.style.format({
            "Baseline": "{:.3f}",
            "Scenario": "{:.3f}"
        }),
        use_container_width=True
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if change < -0.1:

        st.success(
            f"""
            **Model scenario response:** The selected intervention settings
            reduce the model's predicted LST by **{abs(change):.2f} °C**
            relative to the baseline scenario.

            This result should be interpreted as a machine-learning scenario
            response, not as a measured physical cooling effect.
            """
        )

    elif change > 0.1:

        st.warning(
            f"""
            Under these settings, the model predicts an increase of
            **{change:.2f} °C** relative to the baseline scenario.
            """
        )

    else:

        st.info(
            "The selected intervention settings produce little change "
            "in the model prediction."
        )

# ============================================================
# AI RECOMMENDATION
# ============================================================

elif page == "🤖 AI Recommendation":

    st.header("🤖 AI Intervention Recommendation")

    st.write(
        "This section converts the model scenario into an interpretable "
        "planning recommendation."
    )

    # Select hottest hotspot
    hotspot = (
        df.sort_values("LST", ascending=False)
        .iloc[0]
    )

    base_ndvi = float(hotspot["NDVI"])
    base_ndbi = float(hotspot["NDBI"])
    base_ndwi = float(hotspot["NDWI"])
    observed = float(hotspot["LST"])

    baseline = predict_temperature(
        base_ndvi,
        base_ndbi,
        base_ndwi
    )

    # Evaluate individual full-strength scenarios
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

    blue_prediction = predict_temperature(
        base_ndvi,
        base_ndbi,
        ndwi_p75
    )

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
        "Predicted LST": [
            green_prediction,
            blue_prediction,
            built_prediction
        ]
    })

    scenarios["Change"] = (
        scenarios["Predicted LST"] - baseline
    )

    # --------------------------------------------------------
    # CURRENT HOTSPOT
    # --------------------------------------------------------

    st.subheader("🔥 Priority Hotspot")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Observed LST",
            f"{observed:.2f} °C"
        )

    with c2:
        st.metric(
            "AI Baseline",
            f"{baseline:.2f} °C"
        )

    with c3:
        st.metric(
            "Risk",
            heat_category(observed)
        )

    st.divider()

    # --------------------------------------------------------
    # SCENARIO COMPARISON
    # --------------------------------------------------------

    st.subheader("🧊 Intervention Scenario Comparison")

    st.dataframe(
        scenarios.style.format({
            "Predicted LST": "{:.2f}",
            "Change": "{:+.2f}"
        }),
        use_container_width=True
    )

    fig = px.bar(
        scenarios,
        x="Strategy",
        y="Change",
        text_auto=".2f",
        title="Model-Predicted Change Relative to Baseline"
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    best_idx = scenarios["Change"].idxmin()
    best_strategy = scenarios.loc[best_idx, "Strategy"]
    best_change = scenarios.loc[best_idx, "Change"]

    st.subheader("💡 AI Planning Insight")

    st.info(
        f"""
        Under the model's scenario simulation, **{best_strategy}**
        produces the largest reduction in predicted LST among the
        tested intervention scenarios, with a modeled change of
        approximately **{best_change:.2f} °C**.

        This should be treated as a prioritization signal for further
        urban-planning evaluation rather than a guaranteed physical
        cooling effect.
        """
    )

    st.warning(
        "The recommendation is based on the trained model and the "
        "observed feature distribution in the study dataset. Field "
        "validation and physical modeling would be required before "
        "real-world implementation."
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header("📊 Machine Learning Model Performance")

    st.write(
        "HeatShield AI uses a Random Forest Regression model to estimate "
        "land surface temperature from environmental indicators."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "R²",
            f"{metadata['r2']:.4f}"
        )

    with c2:
        st.metric(
            "RMSE",
            f"{metadata['rmse']:.4f} °C"
        )

    with c3:
        st.metric(
            "MAE",
            f"{metadata['mae']:.4f} °C"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.subheader("🌳 Model Configuration")

    details = pd.DataFrame({
        "Parameter": [
            "Model",
            "Target",
            "Features",
            "Trees",
            "Maximum Depth",
            "Training Samples",
            "Test Samples"
        ],
        "Value": [
            metadata["model"],
            metadata["target"],
            ", ".join(metadata["features"]),
            metadata["n_estimators"],
            metadata["max_depth"],
            metadata["training_samples"],
            metadata["test_samples"]
        ]
    })

    st.dataframe(
        details,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("🧠 Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance["Importance (%)"] = (
        importance["Importance"] * 100
    )

    st.dataframe(
        importance.style.format({
            "Importance": "{:.4f}",
            "Importance (%)": "{:.2f}%"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "These metrics describe performance on the held-out test split "
        "used during model development. They do not establish causal "
        "relationships between environmental variables and temperature."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HeatShield AI • Kochi Urban Heat Intelligence • "
    "Satellite-derived indicators + Random Forest"
)