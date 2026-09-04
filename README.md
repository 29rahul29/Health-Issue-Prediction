# Multiple Disease Prediction Syste

A machine learning web application built with Python, Scikit-Learn, and Streamlit to predict the risk of multiple health conditions (Diabetes, Heart Disease, and Parkinson's Disease) based on clinical and biomedical parameters.

---

## Features & Disease Modules

| Disease Module | Machine Learning Model | Key Input Features | Output |
| :--- | :--- | :--- | :--- |
| **Diabetes Prediction** | Support Vector Classifier (`SVC`) with `SimpleImputer` & `StandardScaler` Pipeline | Glucose, Blood Pressure, Insulin, BMI, Age, Pregnancies, Skin Thickness, Diabetes Pedigree Function | Diabetic / Not Diabetic |
| **Heart Disease Prediction** | Logistic Regression | Age, Sex, Chest Pain Type (`cp`), Resting BP (`trestbps`), Cholesterol (`chol`), Fasting Blood Sugar (`fbs`), Resting ECG (`restecg`), Max Heart Rate (`thalach`), Exercise Angina (`exang`), ST Depression (`oldpeak`), Slope, Major Vessels (`ca`), Thalassemia (`thal`) | Heart Disease / No Heart Disease |
| **Parkinson's Disease Prediction** | Support Vector Classifier (`SVC`) | 22 vocal/acoustic signal metrics (Fundamental frequencies `MDVP:Fo/Fhi/Flo`, Jitter metrics, Shimmer metrics, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1/2`, `D2`, `PPE`) | Parkinson's Detected / Healthy |

### Key System Capabilities
- **Pre-trained ML Models & Pipeline**: Integrated end-to-end preprocessing pipeline (`SimpleImputer` + `StandardScaler` + `SVC`) for Diabetes, and serialized classification models for Heart Disease and Parkinson's.
- **Robust Input Validation**: Real-time validation preventing runtime errors on missing or non-numeric inputs.
- **Quick-Fill Sample Data**: Integrated test buttons (`Load Healthy Sample` and `Load High-Risk Sample`) on every module for quick demonstration and testing without manual feature entry.
- **Interactive Multi-Page Interface**: Clean, responsive layout powered by Streamlit and Streamlit Option Menu.

---

## Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Joblib
- **Data Manipulation**: Pandas, NumPy
- **Frontend / Deployment**: Streamlit, Streamlit Option Menu

---

## Project Structure

```
Health Issue prediction/
├── Dataset/
│   ├── diabetes.csv                            # Pima Indians Diabetes dataset
│   ├── heart.csv                               # Cleveland Heart Disease dataset
│   └── parkinsons.csv                          # Oxford Parkinson's Disease dataset
├── Notebook/
│   ├── Multiple disease prediction system - diabetes.ipynb
│   ├── Multiple disease prediction system - heart.ipynb
│   └── Multiple disease prediction system - Parkinsons.ipynb
├── saved_models/
│   ├── diabetes_pipeline.joblib                # Serialized preprocessing + SVC pipeline
│   ├── diabetes_model.sav                      # Legacy serialized SVM model
│   ├── heart_disease_model.sav                 # Serialized Logistic Regression model
│   └── parkinsons_model.sav                    # Serialized SVM model
├── app.py                                      # Streamlit web application
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

---

## Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/29rahul29/Health-Issue-Prediction.git
cd "Health Issue prediction"
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

---

## Medical Disclaimer

> **IMPORTANT**: This application and its underlying machine learning models are designed for **educational, experimental, and portfolio demonstration purposes only**. They do **not** constitute medical advice, diagnosis, or clinical treatment plans. Users must consult licensed medical professionals for any clinical diagnoses and health evaluations.
