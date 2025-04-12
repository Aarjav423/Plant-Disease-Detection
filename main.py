import streamlit as st
import tensorflow as tf
import numpy as np

# Dictionary for disease remedies and solutions
medical_solutions = {
    'Apple___Apple_scab': {
        "Remedy Steps": [
            "1. Remove and destroy infected leaves and twigs.",
            "2. Apply fungicides such as Captan or Mancozeb.",
            "3. Maintain proper spacing for air circulation.",
            "4. Prune trees to improve sunlight penetration."
        ],
        "Recommended Medicines": "Captan, Mancozeb",
        "More Info": "https://extension.umn.edu/plant-diseases/apple-scab",
        "Medicine Info": "https://hort.extension.wisc.edu/articles/applescab/"
    },
    'Apple___Black_rot': {
        "Remedy Steps": [
            "1. Prune and remove infected branches and fruits.",
            "2. Apply fungicides containing Captan or strobilurins.",
            "3. Practice crop rotation and sanitation techniques.",
            "4. Ensure good air circulation and avoid overhead irrigation."
        ],
        "Recommended Medicines": "Captan, Strobilurin Fungicides",
        "More Info": "https://extension.psu.edu/pome-fruit-disease-black-rot-and-frogeye-leaf-spot",
        "Medicine Info": "https://www.gardeningknowhow.com/edible/fruits/apples/black-rot-on-apple-trees.htm"
    },
    'Tomato___Bacterial_spot': {
        "Remedy Steps": [
            "1. Remove infected plants to prevent the spread.",
            "2. Apply copper-based sprays to control bacterial growth.",
            "3. Avoid overhead watering to reduce moisture on leaves.",
            "4. Rotate crops to prevent recurrence."
        ],
        "Recommended Medicines": "Copper Sprays",
        "More Info": "https://extension.umn.edu/disease-management/bacterial-spot-tomato",
        "Medicine Info": "https://www.gardeningknowhow.com/edible/vegetables/tomato/tomato-bacterial-spot.htm"
    }
}

# Severity classification function
def estimate_severity(prediction_confidence):
    if prediction_confidence > 0.8:
        return "Severe"
    elif prediction_confidence > 0.5:
        return "Moderate"
    else:
        return "Mild"

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)  # Return index of max element and confidence score

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Initialize Session State
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "show_remedies" not in st.session_state:
    st.session_state.show_remedies = False
if "show_medicines" not in st.session_state:
    st.session_state.show_medicines = False
if "show_more_info" not in st.session_state:
    st.session_state.show_more_info = False

# Home Page
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE DETECTION SYSTEM USING DEEP LEARNING")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 

    **How It Works:**
    1. **Upload Image**: Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis**: Our system will process the image using deep learning techniques.
    3. **Results**: Get an accurate disease diagnosis along with remedies and recommended medicines.

    🔎 Click on the **Disease Recognition** page in the sidebar to start the analysis!
    """)

# About Project Page
elif app_mode == "About":
    st.header("📜 About the Project")
    st.markdown("""
    This project leverages **Deep Learning and CNNs** to classify plant diseases accurately.
    
    The model has been trained on **over 87,000 images** of healthy and diseased crop leaves, covering **38 different plant diseases**.

    **Dataset Details:**
    - **Train Data**: 70,295 images
    - **Validation Data**: 17,572 images
    - **Test Data**: 33 images
    
    **Objective:**
    - Help farmers and agricultural professionals detect diseases early.
    - Provide **step-by-step remedies** and **medicine recommendations**.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🌱 Disease Recognition System")
    test_image = st.file_uploader("📸 Upload a Plant Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        st.snow()
        st.write("🔎 **Analyzing... Please wait**")
        
        # Model Prediction
        result_index, confidence = model_prediction(test_image)
        
        # Get Disease Name
        class_names = list(medical_solutions.keys())
        disease_detected = class_names[result_index]
        severity = estimate_severity(confidence)
        remedy_steps = medical_solutions[disease_detected]["Remedy Steps"]
        recommended_medicines = medical_solutions[disease_detected]["Recommended Medicines"]
        more_info = medical_solutions[disease_detected]["More Info"]
        medicine_info = medical_solutions[disease_detected]["Medicine Info"]

        # Store results in session state
        st.session_state.prediction_made = True
        st.session_state.disease_detected = disease_detected
        st.session_state.severity = severity
        st.session_state.remedy_steps = remedy_steps
        st.session_state.recommended_medicines = recommended_medicines
        st.session_state.more_info = more_info
        st.session_state.medicine_info = medicine_info

    # Display Results after Prediction
    if st.session_state.prediction_made:
        st.success(f"✅ **Predicted Disease:** {st.session_state.disease_detected}")
        st.info(f"⚠️ **Severity Level:** {st.session_state.severity}")

        # Buttons to show additional info
        if st.button("📖 Show Remedies"):
            st.session_state.show_remedies = not st.session_state.show_remedies
        if st.session_state.show_remedies:
            st.warning(f"🛠 **Step-by-Step Remedy:**")
            for step in st.session_state.remedy_steps:
                st.write(f"- {step}")

        if st.button("💊 Recommended Medicines"):
            st.session_state.show_medicines = not st.session_state.show_medicines
        if st.session_state.show_medicines:
            st.warning(f"💊 **Recommended Medicines:** {st.session_state.recommended_medicines}")
            st.markdown(f"[Click Here for Medicine Info]({st.session_state.medicine_info})")

        if st.button("🔗 More Information"):
            st.session_state.show_more_info = not st.session_state.show_more_info
        if st.session_state.show_more_info:
            st.markdown(f"[Click Here for More Info]({st.session_state.more_info})")
