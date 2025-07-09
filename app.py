import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import joblib 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load model dan scaler yang sudah disimpan
voting_clf_soft = load('obesity_prediction.sav')
scaler = joblib.load('scaler.joblib')

# Fungsi untuk memproses input
def preprocess_input(data):
    # Encoding untuk fitur kategorikal
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC', 'CALC', 'MTRANS']
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CAEC', 'CALC', 'MTRANS']
    
    # Label Encoding untuk fitur kategorikal
    le = LabelEncoder()
    for col in categorical_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    
    # Standard Scaling untuk fitur numerik
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    return data

# Fungsi untuk membuat prediksi
def predict_obesity(data):
    processed_data = preprocess_input(data)
    prediction = voting_clf_soft.predict(processed_data)
    return prediction[0]

# Mapping kelas prediksi ke label yang lebih mudah dibaca
obesity_classes = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Overweight Level I',
    3: 'Overweight Level II',
    4: 'Obesity Level I',
    5: 'Obesity Level II',
    6: 'Obesity Level III'
}

# Tampilan Streamlit
st.title('Obesity Level Prediction')
st.write("""
This application predicts obesity levels based on individual characteristics.
Please enter your information below:
""")

# Form input
with st.form("user_input"):
    st.header("Personal Information")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=14, max_value=100, value=25)
    height = st.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.7, step=0.01)
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
    
    st.header("Eating Habits")
    favc = st.selectbox('Consumption of high caloric food (FAVC)', ['yes', 'no'])
    fcvc = st.slider('Eat vegetables in your meals (FCVC)', 1, 3, 2)
    ncp = st.slider('Number of main meals (NCP)', 1, 4, 3)
    caec = st.selectbox('Consumption of food between meals (CAEC)', 
                       ['no', 'Sometimes', 'Frequently', 'Always'])
    scc = st.selectbox('Calories consumption monitoring (SCC)', ['yes', 'no'])
    calc = st.selectbox('Consumption of alcohol (CALC)', 
                        ['no', 'Sometimes', 'Frequently', 'Always'])
    
    st.header("Physical Activity")
    mtrans = st.selectbox('Transportation used (MTRANS)', 
                          ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])
    
    submitted = st.form_submit_button("Predict Obesity Level")

# Ketika form disubmit
if submitted:
    # Membuat dataframe dari input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SCC': [scc],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    
    # Melakukan prediksi
    try:
        prediction = predict_obesity(input_data)
        result = obesity_classes[prediction]
        
        st.success(f"Predicted Obesity Level: {result}")
        
        # Menampilkan penjelasan hasil
        st.subheader("Interpretation:")
        if prediction == 0:
            st.write("You are underweight. Consider consulting a nutritionist for a balanced diet plan.")
        elif prediction == 1:
            st.write("Your weight is normal. Maintain your healthy lifestyle!")
        elif prediction == 2:
            st.write("You are Class I overweight. Consider increasing physical activity.")  
        elif prediction == 3:
            st.write("You are Class II overweight. A healthier diet and more exercise would be beneficial.")  
        elif prediction == 4:
            st.write("You have Class I obesity. Consider consulting a healthcare professional.")
        elif prediction == 5:
            st.write("You have Class II obesity. Medical advice is recommended.")
        elif prediction == 6:
            st.write("You have Class III obesity. Please consult a healthcare professional immediately.")
            
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Informasi tambahan
st.sidebar.header("About")
st.sidebar.info("""
This app predicts obesity levels based on individual characteristics using an ensemble machine learning model.
The model combines Decision Tree, Random Forest, and Logistic Regression with soft voting.
""")

st.sidebar.header("Input Guide")
st.sidebar.write("""
- **FCVC**: Eeating vegetables in your meals per day (1-3 scale)
- **NCP**: Number of main meals per day (1-4)
- **CAEC**: Frequency of eating between meals
- **CALC**: Frequency of alcohol consumption
- **MTRANS**: Primary mode of transportation
""")