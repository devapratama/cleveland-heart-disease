import streamlit as st
import pickle
import numpy as np

# Load your trained Random Forest model
model = pickle.load(open('rf.pkl', 'rb'))

# Load StandardScaler from pickle file
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a function to preprocess user input
def preprocess_input(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Standardize the input data using the loaded scaler
    standardized_data = scaler.transform(input_data_as_numpy_array)

    return standardized_data

st.set_page_config(
  page_title = "Cleveland Heart Disease",
  page_icon = ":heart:"
)

# Streamlit application layout
st.title('Heart Disease Prediction App')
st.markdown("By: ")
st.markdown("This tool predicts the likelihood of heart disease based on medical diagnostic measures.")

# About Heart Disease
st.markdown("### About Heart Disease")
st.write("Heart disease, also known as cardiovascular disease (CVD), encompasses a broad range of conditions that affect the heart and blood vessels. It is a leading cause of mortality worldwide, with several types and stages. Some common types of CVD include coronary artery disease (CAD), heart failure, and arrhythmias.")
st.write("Understanding the risk factors and early detection of heart disease are critical for prevention and timely intervention. This tool employs machine learning to assess the probability of heart disease based on a set of clinical and diagnostic parameters.")

# Input fields for user data
st.markdown("### Patient Information")

# Age Input
age = st.number_input('Age (in years)', min_value=0, max_value=120, value=30, 
                      help='Enter the age of the patient in years.')

# Sex Input
sex = st.radio('Sex', options=[('Female', 0), ('Male', 1)], format_func=lambda x: x[0], 
               help='Select the sex of the patient.')

# Chest Pain Type Input
cp = st.selectbox('Chest Pain Type', options=[('Typical Angina', 1), ('Atypical Angina', 2), 
                                               ('Non-Anginal Pain', 3), ('Asymptomatic', 4)], format_func=lambda x: x[0], 
                                               help='Select the type of chest pain experienced by the patient:\n'
                                               '- Typical Angina: A type of chest pain related to heart.\n'
                                               '- Atypical Angina: Chest pain not related to heart.\n'
                                               '- Non-Anginal Pain: Non-specific chest pain.\n'
                                               '- Asymptomatic: No chest pain.')

# Resting Blood Pressure Input
trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=0, value=120,
                           help='Enter the blood pressure reading of the patient on hospital admission.')

# Serum Cholestoral Input
chol = st.number_input('Serum Cholestoral (in mg/dl)', min_value=0, value=200, 
                       help='Enter the patient’s cholesterol measurement in milligrams per deciliter.')

# Fasting Blood Sugar Input
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0],
               help='Indicate whether the patient’s fasting blood sugar is more than 120 mg/dl:\n'
               '- No: Fasting blood sugar is not more than 120 mg/dl.\n'
               '- Yes: Fasting blood sugar is more than 120 mg/dl.')

# Resting Electrocardiographic Results Input
restecg = st.selectbox('Resting Electrocardiographic Results', options=[('Normal', 0), ('ST-T Wave Abnormality', 1), 
                                                                        ('Left Ventricular Hypertrophy', 2)], format_func=lambda x: x[0], 
                                                                        help='Select the results of electrocardiogram measurements at rest:\n'
                                                                        '- Normal: Normal ECG results.\n'
                                                                        '- ST-T Wave Abnormality: Abnormal ECG results related to the ST-T wave.\n'
                                                                        '- Left Ventricular Hypertrophy: Abnormal ECG results indicating left ventricular hypertrophy.')

# Maximum Heart Rate Achieved Input
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, value=100, 
                         help='Enter the highest heart rate the patient achieved during a stress test.')

# Exercise Induced Angina Input
exang = st.radio('Exercise Induced Angina', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0],
                 help='Indicate whether the patient experienced chest pain during exercise:\n'
                 '- No: No chest pain during exercise.\n'
                 '- Yes: Chest pain experienced during exercise.')

# ST Depression Induced by Exercise Relative to Rest Input
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                    help='This is a measure of heart stress during exercise relative to rest.')

# Slope of the Peak Exercise ST Segment Input
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[('Upsloping', 1), ('Flat', 2), 
                                                                       ('Downsloping', 3)], format_func=lambda x: x[0], 
                                                                       help='Select the slope of the peak exercise ST segment, an ECG reading related to heart function:\n'
                                                                       '- Upsloping: Upsloping ST segment during exercise.\n'
                                                                       '- Flat: Flat ST segment during exercise.\n'
                                                                       '- Downsloping: Downsloping ST segment during exercise.')

# Number of Major Vessels Colored by Flourosopy Input
ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=3, value=0, 
               help='Enter the number of major blood vessels supplying blood to the heart as seen through fluoroscopy.')

# Thalassemia Input
thal = st.selectbox('Thalassemia', options=[('Normal', 3), ('Fixed Defect', 6), 
                                             ('Reversible Defect', 7)], format_func=lambda x: x[0], 
                                             help='Select the type of Thalassemia, measured via a blood test:\n'
                                             '- Normal: No Thalassemia detected.\n'
                                             '- Fixed Defect: Fixed Thalassemia defect detected.\n'
                                             '- Reversible Defect: Reversible Thalassemia defect detected.')

st.markdown("### Prediction")

# Predict button
if st.button('Predict Heart Disease'):
    input_data = [age, sex[1], cp[1], trestbps, chol, fbs[1], restecg[1], thalach, exang[1], oldpeak, slope[1], ca, thal[1]]
    preprocessed_input = preprocess_input(input_data)
    prediction = model.predict(preprocessed_input)

    # Map numerical prediction to clinical outcome
    prediction_map = {0: 'Less than 50% Diameter Narrowing - No Heart Disease', 
                      1: 'Greater than 50% Diameter Narrowing - Stage 1', 
                      2: 'Greater than 50% Diameter Narrowing - Stage 2', 
                      3: 'Greater than 50% Diameter Narrowing - Stage 3', 
                      4: 'Greater than 50% Diameter Narrowing - Stage 4'}

    # Set the color based on prediction
    prediction_color = {
        0: 'green',  # No Heart Disease - Green
        1: 'yellow',  # Stage 1 Heart Disease - Yellow
        2: 'orange',  # Stage 2 Heart Disease - Orange
        3: 'red',  # Stage 3 Heart Disease - Red
        4: 'purple'  # Stage 4 Heart Disease - Purple
    }

    prediction_text = f'<span style="font-size:24px;">Prediction: <span style="color:{prediction_color[prediction[0]]};">{prediction_map[prediction[0]]}</span></span>'

    # Display the prediction with the specified color and larger font size
    st.markdown(prediction_text, unsafe_allow_html=True)

# Prediction Details
st.markdown("### Prediction Details")
st.write("The prediction provided by this tool is based on a Random Forest model. A Random Forest is a machine learning algorithm that combines the input data to estimate the probability of heart disease.")
st.write("The model categorizes the likelihood of heart disease into the following stages based on the probability:")
st.write("- **Stage 0: No Heart Disease**: This category represents individuals with a low probability of having heart disease, indicating a healthy heart.")
st.write("- **Stage 1: Mild Heart Disease**: Individuals in this category have a higher probability of having heart disease, typically associated with mild symptoms or early stages of the condition.")
st.write("- **Stage 2: Moderate Heart Disease**: This stage indicates a moderate likelihood of heart disease, often requiring medical attention and further evaluation.")
st.write("- **Stage 3: Severe Heart Disease**: Individuals in this category have a high probability of severe heart disease, which may necessitate immediate medical intervention.")
st.write("- **Stage 4: Critical Heart Disease**: This stage represents a critical likelihood of heart disease, indicating a severe condition that requires immediate medical attention and treatment.")

# Disclaimer
st.info("Disclaimer: The results presented by this tool are based on the input data and should not be taken as a substitute for professional medical advice. If you're experiencing symptoms or have concerns about your health, please consult a healthcare provider for an accurate diagnosis.")

# References
st.markdown("### References")
st.write("For comprehensive information about heart disease, please refer to the following reputable sources:")
st.write("- [American Heart Association](https://www.heart.org/)")
st.write("- [Mayo Clinic](https://www.mayoclinic.org/)")
st.write("- [MedlinePlus - NIH](https://medlineplus.gov/heartdiseases.html)")

# Tambahkan kode CSS untuk mode gelap
st.markdown("""
    <style>
    /* Dark Mode */
    .reportview-container.darkMode {
        background-color: #2C3E50;
    }
    .stTextInput.darkMode, .stSelectbox.darkMode, .stNumberInput.darkMode, .stRadio.darkMode, .stSlider.darkMode {
        border: 1px solid #34495E;
        background-color: #34495E;
    }
    .stTextInput.darkMode > label, .stSelectbox.darkMode > label, .stNumberInput.darkMode > label, .stRadio.darkMode > label, .stSlider.darkMode > label {
        color: #ECF0F1;
    }
    .stTextInput.darkMode > div > div > input, .stSelectbox.darkMode > select, .stNumberInput.darkMode > div > div > input {
        color: #ECF0F1;
    }
    .stRadio.darkMode > div > div > label {
        background-color: #34495E;
        border-color: #3498DB;
    }
    .stRadio.darkMode > div > div > label:hover {
        background: #3B4E61;
        border-color: #E74C3C;
    }
    .stSlider.darkMode > div > div {
        background-color: #34495E;
    }
    .stSlider.darkMode > div > div > div {
        background-color: #3498DB;
    }
    .stMarkdown.darkMode {
        color: #ECF0F1;
    }
    .stAlert.darkMode {
        background-color: #E74C3C;
    }
    </style>
""", unsafe_allow_html=True)

# Tambahkan CSS untuk mode terang
st.markdown("""
    <style>
    /* Light Mode */
    .reportview-container.lightMode {
        background-color: #FFFFFF;
    }
    .stTextInput.lightMode, .stSelectbox.lightMode, .stNumberInput.lightMode, .stRadio.lightMode, .stSlider.lightMode {
        border: 1px solid #ced4da;
        background-color: #F4F6F6;
    }
    .stTextInput.lightMode > label, .stSelectbox.lightMode > label, .stNumberInput.lightMode > label, .stRadio.lightMode > label, .stSlider.lightMode > label {
        color: #2C3E50;
    }
    .stTextInput.lightMode > div > div > input, .stSelectbox.lightMode > select, .stNumberInput.lightMode > div > div > input {
        color: #2C3E50;
    }
    .stRadio.lightMode > div > div > label {
        background-color: #F4F6F6;
        border-color: #ced4da;
    }
    .stRadio.lightMode > div > div > label:hover {
        background: #D6DBDF;
        border-color: #BDC3C7;
    }
    .stSlider.lightMode > div > div {
        background-color: #ECF0F1;
    }
    .stSlider.lightMode > div > div > div {
        background-color: #3498DB;
    }
    .stMarkdown.lightMode {
        color: #2C3E50;
    }
    .stAlert.lightMode {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
    }
    </style>
""", unsafe_allow_html=True)
