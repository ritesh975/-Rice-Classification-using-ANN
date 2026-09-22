import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Rice Classification ANN",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f5fff5, #ffffff);
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #1b5e20;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.95);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.12);
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 700;
}

.feature-title {
    font-size: 22px;
    font-weight: 700;
    color: #2e7d32;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "rice_ann_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_resources():

    # Load ANN model
    model = load_model(
        MODEL_PATH,
        compile=False
    )

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# =========================================================
# LOAD RESOURCES SAFELY
# =========================================================

try:

    model, scaler = load_resources()

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.info(
        "Please make sure that rice_ann_model.keras and scaler.pkl "
        "are present in the same folder as app.py."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🌾 Rice Classification using ANN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Neural Network based Rice Grain Classification System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌾 Rice ANN")

    st.write(
        """
        This application uses an Artificial Neural Network
        to classify rice grains based on their physical
        characteristics.
        """
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("Artificial Neural Network")

    st.markdown("### 📊 Features")

    st.write("10 input features")

    st.markdown("### ⚙️ Preprocessing")

    st.write("Standard Scaler")

    st.divider()

    st.caption("Rice Classification ANN Project")


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="feature-title">📊 Enter Rice Grain Features</div>',
    unsafe_allow_html=True
)

st.write("Enter the physical characteristics of the rice grain.")

col1, col2 = st.columns(2)


# =========================================================
# FEATURE INPUTS
# =========================================================

with col1:

    area = st.number_input(
        "Area",
        min_value=0.0,
        value=5000.0,
        step=10.0
    )

    major_axis_length = st.number_input(
        "Major Axis Length",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    minor_axis_length = st.number_input(
        "Minor Axis Length",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    eccentricity = st.number_input(
        "Eccentricity",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    convex_area = st.number_input(
        "Convex Area",
        min_value=0.0,
        value=5100.0,
        step=10.0
    )


with col2:

    equiv_diameter = st.number_input(
        "Equivalent Diameter",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

    extent = st.number_input(
        "Extent",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    perimeter = st.number_input(
        "Perimeter",
        min_value=0.0,
        value=300.0,
        step=1.0
    )

    roundness = st.number_input(
        "Roundness",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    aspect_ratio = st.number_input(
        "Aspect Ratio",
        min_value=0.0,
        value=1.5,
        step=0.01
    )


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("### 🔮 Classification")

predict_button = st.button(
    "🚀 Predict Rice Class",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # Input feature order MUST match training order
        # -------------------------------------------------

        input_data = np.array([[
            area,
            major_axis_length,
            minor_axis_length,
            eccentricity,
            convex_area,
            equiv_diameter,
            extent,
            perimeter,
            roundness,
            aspect_ratio
        ]])

        # -------------------------------------------------
        # Scale input
        # -------------------------------------------------

        scaled_data = scaler.transform(input_data)

        # -------------------------------------------------
        # ANN Prediction
        # -------------------------------------------------

        prediction = model.predict(
            scaled_data,
            verbose=0
        )

        probability = float(prediction[0][0])

        # -------------------------------------------------
        # Binary Classification
        # -------------------------------------------------

        if probability >= 0.5:

            predicted_class = 1
            confidence = probability * 100

        else:

            predicted_class = 0
            confidence = (1 - probability) * 100


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown("## 🌾 Prediction Result")

        if predicted_class == 1:

            st.success(
                "### Predicted Class: 1"
            )

        else:

            st.info(
                "### Predicted Class: 0"
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(max(confidence / 100, 0.0), 1.0)
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.markdown("### 📋 Input Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(f"**Area:** {area}")
            st.write(f"**Major Axis Length:** {major_axis_length}")
            st.write(f"**Minor Axis Length:** {minor_axis_length}")
            st.write(f"**Eccentricity:** {eccentricity}")
            st.write(f"**Convex Area:** {convex_area}")

        with summary_col2:

            st.write(f"**Equivalent Diameter:** {equiv_diameter}")
            st.write(f"**Extent:** {extent}")
            st.write(f"**Perimeter:** {perimeter}")
            st.write(f"**Roundness:** {roundness}")
            st.write(f"**Aspect Ratio:** {aspect_ratio}")


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))
