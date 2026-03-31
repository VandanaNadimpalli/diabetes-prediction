# app.py
# ============================================
# PART 1: IMPORT LIBRARIES
# ============================================

import streamlit as st      # For creating web interface
import joblib               # To load the saved model
import numpy as np          # For calculations
import pandas as pd         # For displaying data

# WHAT THIS PART DOES:
# - streamlit: Makes it easy to create websites with Python
# - joblib: Loads our trained model file

# ============================================
# PART 2: SET UP THE PAGE
# ============================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("""
This app uses **Machine Learning** to predict diabetes risk based on patient health metrics.
Enter the values below and click 'Predict' to see the result.
""")

# WHAT THIS PART DOES:
# - Creates the webpage title and layout
# - "wide" makes it use full screen width

# ============================================
# PART 3: SIDEBAR INFORMATION
# ============================================

with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model**: Logistic Regression  
    **Accuracy**: ~97%  
    **Features**: 8 health metrics  
    **Dataset**: Pima Indians Diabetes
    """)
    
    st.header("📋 Instructions")
    st.markdown("""
    1. Enter patient details
    2. Click 'Predict'
    3. View instant results
    """)

# WHAT THIS PART DOES:
# - Adds a sidebar with information
# - Users can see what the app does

# ============================================
# PART 4: LOAD THE MODEL
# ============================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        st.stop()

model = load_model()

# WHAT THIS PART DOES:
# - Loads our saved model file
# - @st.cache_resource means it loads only once (faster!)
# - Shows error if model file missing

# ============================================
# PART 5: CREATE INPUT FORM
# ============================================

st.header("📝 Patient Health Metrics")

with st.form("prediction_form"):
    # First row - 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input(
            "Pregnancies", 
            min_value=0, max_value=20, value=1,
            help="Number of pregnancies"
        )
    
    with col2:
        glucose = st.number_input(
            "Glucose (mg/dL)", 
            min_value=0, max_value=300, value=100,
            help="Plasma glucose concentration"
        )
    
    with col3:
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)", 
            min_value=0, max_value=200, value=70,
            help="Diastolic blood pressure"
        )
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", 
            min_value=0, max_value=100, value=20,
            help="Triceps skin fold thickness"
        )
    
    with col5:
        insulin = st.number_input(
            "Insulin (μU/mL)", 
            min_value=0, max_value=900, value=80,
            help="2-Hour serum insulin"
        )
    
    with col6:
        bmi = st.number_input(
            "BMI", 
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Body mass index"
        )
    
    # Third row
    col7, col8, _ = st.columns(3)
    
    with col7:
        dpf = st.number_input(
            "Diabetes Pedigree", 
            min_value=0.0, max_value=3.0, value=0.5, step=0.01,
            help="Diabetes pedigree function"
        )
    
    with col8:
        age = st.number_input(
            "Age", 
            min_value=0, max_value=120, value=30,
            help="Age in years"
        )
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Diabetes Risk", use_container_width=True)

# WHAT THIS PART DOES:
# - Creates input fields for all 8 measurements
# - st.number_input creates boxes where users type numbers
# - min/max values prevent unrealistic entries
# - help text shows when hovering over input

# ============================================
# PART 6: MAKE PREDICTION
# ============================================

if submitted:
    # Create array of input values
    features = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Show results
    st.header("🎯 Prediction Result")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.error("### 🔴 High Risk of Diabetes")
            st.markdown("""
            **Recommendations:**
            - Consult a healthcare provider
            - Monitor blood sugar levels
            - Maintain healthy diet
            - Regular exercise
            """)
        else:
            st.success("### 🟢 Low Risk of Diabetes")
            st.markdown("""
            **Recommendations:**
            - Maintain healthy lifestyle
            - Regular checkups
            - Balanced diet
            - Stay active
            """)
    
    with result_col2:
        risk_percentage = probability[1] * 100
        st.metric("Diabetes Risk Probability", f"{risk_percentage:.1f}%")
        
        # Show entered values
        st.subheader("📋 Entered Values")
        values_df = pd.DataFrame({
            'Feature': ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age]
        })
        st.dataframe(values_df, use_container_width=True, hide_index=True)

# WHAT THIS PART DOES:
# - Takes user inputs and feeds to model
# - Shows prediction (diabetes or not)
# - Shows risk percentage
# - Gives health recommendations
# - Displays entered values for verification

# ============================================
# PART 7: FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
    <p style='color: gray; font-size: 0.8em;'>© 2026 Diabetes Prediction App</p>
</div>
""", unsafe_allow_html=True)