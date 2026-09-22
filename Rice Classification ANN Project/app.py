import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Rice Classification",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

@st.cache_resource
def load_resources():

    model = load_model("rice_ann_model.keras")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_resources()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🌾 Rice Classification using ANN")

st.write(
    "Enter the rice grain characteristics below "
    "to predict the rice class."
)

st.divider()


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("📊 Enter Rice Features")


col1, col2 = st.columns(2)


with col1:

    Area = st.number_input(
        "Area",
        min_value=0.0,
        value=3000.0
    )

    MajorAxisLength = st.number_input(
        "Major Axis Length",
        min_value=0.0,
        value=75.0
    )

    MinorAxisLength = st.number_input(
        "Minor Axis Length",
        min_value=0.0,
        value=50.0
    )

    Eccentricity = st.number_input(
        "Eccentricity",
        min_value=0.0,
        max_value=1.0,
        value=0.75
    )

    ConvexArea = st.number_input(
        "Convex Area",
        min_value=0.0,
        value=3100.0
    )


with col2:

    EquivDiameter = st.number_input(
        "Equivalent Diameter",
        min_value=0.0,
        value=60.0
    )

    Extent = st.number_input(
        "Extent",
        min_value=0.0,
        max_value=1.0,
        value=0.75
    )

    Perimeter = st.number_input(
        "Perimeter",
        min_value=0.0,
        value=220.0
    )

    Roundness = st.number_input(
        "Roundness",
        min_value=0.0,
        max_value=1.0,
        value=0.80
    )

    AspectRation = st.number_input(
        "Aspect Ratio",
        min_value=0.0,
        value=1.5
    )


st.divider()


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button(
    "🔍 Predict Rice Class",
    use_container_width=True
):

    # Create input array
    input_data = np.array([[
        Area,
        MajorAxisLength,
        MinorAxisLength,
        Eccentricity,
        ConvexArea,
        EquivDiameter,
        Extent,
        Perimeter,
        Roundness,
        AspectRation
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction probability
    prediction_probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    # Binary classification
    if prediction_probability >= 0.5:

        prediction = 1
        class_name = "Class 1"

    else:

        prediction = 0
        class_name = "Class 0"


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.success(
            f"Predicted Rice Class: {class_name}"
        )

    else:

        st.info(
            f"Predicted Rice Class: {class_name}"
        )


    # Probability

    st.write(
        f"Prediction Probability: "
        f"**{prediction_probability:.2%}**"
    )


    # Progress bar

    st.progress(
        float(prediction_probability)
    )


    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    st.subheader("📋 Input Features")

    input_display = {

        "Area": Area,
        "Major Axis Length": MajorAxisLength,
        "Minor Axis Length": MinorAxisLength,
        "Eccentricity": Eccentricity,
        "Convex Area": ConvexArea,
        "Equivalent Diameter": EquivDiameter,
        "Extent": Extent,
        "Perimeter": Perimeter,
        "Roundness": Roundness,
        "Aspect Ratio": AspectRation

    }

    st.dataframe(
        input_display,
        use_container_width=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Rice Classification | Deep Learning ANN | Streamlit"
)