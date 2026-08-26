import os
import pickle
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# getting the working directory of app.py
working_dir = os.path.dirname(os.path.abspath(__file__))


# loading the saved models with caching
@st.cache_resource
def load_models():
    # Diabetes & Parkinson's use full Scikit-Learn pipelines
    diabetes_pipeline_path = os.path.join(working_dir, 'saved_models', 'diabetes_pipeline.joblib')
    heart_model_path = os.path.join(working_dir, 'saved_models', 'heart_disease_model.sav')
    parkinsons_pipeline_path = os.path.join(working_dir, 'saved_models', 'parkinsons_pipeline.joblib')

    diabetes_pipeline = joblib.load(diabetes_pipeline_path)
    heart_disease_model = pickle.load(open(heart_model_path, 'rb'))
    parkinsons_pipeline = joblib.load(parkinsons_pipeline_path)

    return diabetes_pipeline, heart_disease_model, parkinsons_pipeline


diabetes_model, heart_disease_model, parkinsons_model = load_models()


with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )


# Helper function for safe input validation
def parse_numeric_inputs(inputs, field_names):
    """Validates and converts input strings into a list of floats.
    Returns (cleaned_floats, error_message).
    """
    empty_fields = [name for name, val in zip(field_names, inputs) if str(val).strip() == '']
    if empty_fields:
        return None, f"Please fill in all fields. Missing: {', '.join(empty_fields)}"

    cleaned = []
    invalid_fields = []
    for name, val in zip(field_names, inputs):
        try:
            cleaned.append(float(str(val).strip()))
        except ValueError:
            invalid_fields.append(name)

    if invalid_fields:
        return None, f"Invalid numeric value in: {', '.join(invalid_fields)}"

    return cleaned, None


# Helper callback to populate input fields with sample values
def load_sample_values(sample_dict):
    for key, value in sample_dict.items():
        st.session_state[key] = str(value)


# Realistic Sample Data from Datasets
SAMPLES = {
    'diabetes_healthy': {
        'diab_pregnancies': '1',
        'diab_glucose': '85',
        'diab_bp': '66',
        'diab_skin': '29',
        'diab_insulin': '26',
        'diab_bmi': '26.6',
        'diab_dpf': '0.351',
        'diab_age': '31'
    },
    'diabetes_risk': {
        'diab_pregnancies': '6',
        'diab_glucose': '148',
        'diab_bp': '72',
        'diab_skin': '35',
        'diab_insulin': '168',
        'diab_bmi': '33.6',
        'diab_dpf': '0.627',
        'diab_age': '50'
    },
    'heart_healthy': {
        'heart_age': '67',
        'heart_sex': '1',
        'heart_cp': '0',
        'heart_trestbps': '160',
        'heart_chol': '286',
        'heart_fbs': '0',
        'heart_restecg': '0',
        'heart_thalach': '108',
        'heart_exang': '1',
        'heart_oldpeak': '1.5',
        'heart_slope': '1',
        'heart_ca': '3',
        'heart_thal': '2'
    },
    'heart_risk': {
        'heart_age': '63',
        'heart_sex': '1',
        'heart_cp': '3',
        'heart_trestbps': '145',
        'heart_chol': '233',
        'heart_fbs': '1',
        'heart_restecg': '0',
        'heart_thalach': '150',
        'heart_exang': '0',
        'heart_oldpeak': '2.3',
        'heart_slope': '0',
        'heart_ca': '0',
        'heart_thal': '1'
    },
    'parkinsons_healthy': {
        'park_fo': '197.076',
        'park_fhi': '206.896',
        'park_flo': '192.055',
        'park_jitter_pct': '0.00289',
        'park_jitter_abs': '0.00001',
        'park_rap': '0.00166',
        'park_ppq': '0.00168',
        'park_ddp': '0.00498',
        'park_shimmer': '0.01098',
        'park_shimmer_db': '0.097',
        'park_apq3': '0.00563',
        'park_apq5': '0.0068',
        'park_apq': '0.00802',
        'park_dda': '0.01689',
        'park_nhr': '0.00339',
        'park_hnr': '26.775',
        'park_rpde': '0.422229',
        'park_dfa': '0.741367',
        'park_spread1': '-7.3483',
        'park_spread2': '0.177551',
        'park_d2': '1.743867',
        'park_ppe': '0.085569'
    },
    'parkinsons_risk': {
        'park_fo': '119.992',
        'park_fhi': '157.302',
        'park_flo': '74.997',
        'park_jitter_pct': '0.00784',
        'park_jitter_abs': '0.00007',
        'park_rap': '0.0037',
        'park_ppq': '0.00554',
        'park_ddp': '0.01109',
        'park_shimmer': '0.04374',
        'park_shimmer_db': '0.426',
        'park_apq3': '0.02182',
        'park_apq5': '0.0313',
        'park_apq': '0.02971',
        'park_dda': '0.06545',
        'park_nhr': '0.02211',
        'park_hnr': '21.033',
        'park_rpde': '0.414783',
        'park_dfa': '0.815285',
        'park_spread1': '-4.813031',
        'park_spread2': '0.266482',
        'park_d2': '2.301442',
        'park_ppe': '0.284654'
    }
}


# ==========================================
# 1. Diabetes Prediction Page
# ==========================================
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        st.button('Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['diabetes_healthy'],), key='btn_diab_healthy')
    with col_btn2:
        st.button('Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['diabetes_risk'],), key='btn_diab_risk')

    with st.form(key='diabetes_form', border=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', key='diab_pregnancies')

        with col2:
            Glucose = st.text_input('Glucose Level', key='diab_glucose')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value', key='diab_bp')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value', key='diab_skin')

        with col2:
            Insulin = st.text_input('Insulin Level', key='diab_insulin')

        with col3:
            BMI = st.text_input('BMI value', key='diab_bmi')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', key='diab_dpf')

        with col2:
            Age = st.text_input('Age of the Person', key='diab_age')

        submitted = st.form_submit_button('Diabetes Test Result')

    if submitted:
        field_names = [
            'Number of Pregnancies', 'Glucose Level', 'Blood Pressure',
            'Skin Thickness', 'Insulin Level', 'BMI',
            'Diabetes Pedigree Function', 'Age'
        ]
        raw_inputs = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        cleaned_inputs, error_msg = parse_numeric_inputs(raw_inputs, field_names)

        if error_msg:
            st.error(error_msg)
        else:
            feature_cols = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            input_df = pd.DataFrame([cleaned_inputs], columns=feature_cols)
            diab_prediction = diabetes_model.predict(input_df)

            if diab_prediction[0] == 1:
                st.warning('The person is diabetic')
            else:
                st.success('The person is not diabetic')


# ==========================================
# 2. Heart Disease Prediction Page
# ==========================================
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        st.button('Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['heart_healthy'],), key='btn_heart_healthy')
    with col_btn2:
        st.button('Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['heart_risk'],), key='btn_heart_risk')

    with st.form(key='heart_form', border=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age', key='heart_age')

        with col2:
            sex = st.text_input('Sex (1 = male; 0 = female)', key='heart_sex')

        with col3:
            cp = st.text_input('Chest Pain types (0, 1, 2, 3)', key='heart_cp')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure', key='heart_trestbps')

        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl', key='heart_chol')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', key='heart_fbs')

        with col1:
            restecg = st.text_input('Resting Electrocardiographic results (0, 1, 2)', key='heart_restecg')

        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved', key='heart_thalach')

        with col3:
            exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)', key='heart_exang')

        with col1:
            oldpeak = st.text_input('ST depression induced by exercise', key='heart_oldpeak')

        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment (0, 1, 2)', key='heart_slope')

        with col3:
            ca = st.text_input('Major vessels colored by flourosopy (0-3)', key='heart_ca')

        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', key='heart_thal')

        submitted = st.form_submit_button('Heart Disease Test Result')

    if submitted:
        field_names = [
            'Age', 'Sex', 'Chest Pain types', 'Resting Blood Pressure',
            'Serum Cholestoral', 'Fasting Blood Sugar', 'Resting ECG',
            'Max Heart Rate', 'Exercise Induced Angina', 'ST depression (oldpeak)',
            'Slope', 'Major vessels (ca)', 'Thal'
        ]
        raw_inputs = [
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]

        cleaned_inputs, error_msg = parse_numeric_inputs(raw_inputs, field_names)

        if error_msg:
            st.error(error_msg)
        else:
            feature_cols = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            input_df = pd.DataFrame([cleaned_inputs], columns=feature_cols)
            heart_prediction = heart_disease_model.predict(input_df)

            if heart_prediction[0] == 1:
                st.warning('The person is having heart disease')
            else:
                st.success('The person does not have any heart disease')


# ==========================================
# 3. Parkinson's Prediction Page
# ==========================================
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        st.button('Load Healthy Sample', on_click=load_sample_values, args=(SAMPLES['parkinsons_healthy'],), key='btn_park_healthy')
    with col_btn2:
        st.button('Load High-Risk Sample', on_click=load_sample_values, args=(SAMPLES['parkinsons_risk'],), key='btn_park_risk')

    with st.form(key='parkinsons_form', border=False):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)', key='park_fo')

        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)', key='park_fhi')

        with col3:
            flo = st.text_input('MDVP:Flo(Hz)', key='park_flo')

        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)', key='park_jitter_pct')

        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', key='park_jitter_abs')

        with col1:
            RAP = st.text_input('MDVP:RAP', key='park_rap')

        with col2:
            PPQ = st.text_input('MDVP:PPQ', key='park_ppq')

        with col3:
            DDP = st.text_input('Jitter:DDP', key='park_ddp')

        with col4:
            Shimmer = st.text_input('MDVP:Shimmer', key='park_shimmer')

        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', key='park_shimmer_db')

        with col1:
            APQ3 = st.text_input('Shimmer:APQ3', key='park_apq3')

        with col2:
            APQ5 = st.text_input('Shimmer:APQ5', key='park_apq5')

        with col3:
            APQ = st.text_input('MDVP:APQ', key='park_apq')

        with col4:
            DDA = st.text_input('Shimmer:DDA', key='park_dda')

        with col5:
            NHR = st.text_input('NHR', key='park_nhr')

        with col1:
            HNR = st.text_input('HNR', key='park_hnr')

        with col2:
            RPDE = st.text_input('RPDE', key='park_rpde')

        with col3:
            DFA = st.text_input('DFA', key='park_dfa')

        with col4:
            spread1 = st.text_input('spread1', key='park_spread1')

        with col5:
            spread2 = st.text_input('spread2', key='park_spread2')

        with col1:
            D2 = st.text_input('D2', key='park_d2')

        with col2:
            PPE = st.text_input('PPE', key='park_ppe')

        submitted = st.form_submit_button("Parkinson's Test Result")

    if submitted:
        field_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
            'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        raw_inputs = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs,
            RAP, PPQ, DDP, Shimmer, Shimmer_dB,
            APQ3, APQ5, APQ, DDA, NHR,
            HNR, RPDE, DFA, spread1, spread2, D2, PPE
        ]

        cleaned_inputs, error_msg = parse_numeric_inputs(raw_inputs, field_names)

        if error_msg:
            st.error(error_msg)
        else:
            feature_cols = [
                'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
                'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
            ]
            input_df = pd.DataFrame([cleaned_inputs], columns=feature_cols)
            parkinsons_prediction = parkinsons_model.predict(input_df)

            if parkinsons_prediction[0] == 1:
                st.warning("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")