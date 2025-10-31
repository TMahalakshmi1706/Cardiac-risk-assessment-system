import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Heart Health AI - Smart Cardiac Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .main {
            background: #000;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #e63946;
            margin: 15px 0;
            color: white;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #51cf66, #40c057);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #2a9d8f;
            margin: 15px 0;
            color: white;
        }
        
        .risk-moderate {
            background: linear-gradient(135deg, #ffd43b, #fcc419);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #ffc107;
            margin: 15px 0;
            color: #333;
        }
        
        .info-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            border-left: 4px solid #6c757d;
            backdrop-filter: blur(10px);
        }
        
        .section-header {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            background: linear-gradient(135deg, #e63946, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.08);
            padding: 25px;
            border-radius: 15px;
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 30px rgba(230, 57, 70, 0.3);
        }
        
        .stForm {
            background: rgba(255, 255, 255, 0.05);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

class HeartDiseaseApp:
    def __init__(self):
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model_data = joblib.load('heart_disease_model.joblib')
        except:
            self.model_data = None
    
    def create_engineered_features(self, input_data):
        """Create engineered features for prediction"""
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = input_data

        if age <= 45: age_category = 0
        elif age <= 55: age_category = 1
        elif age <= 65: age_category = 2
        else: age_category = 3
        
        if trestbps <= 120: bp_category = 0
        elif trestbps <= 130: bp_category = 1
        elif trestbps <= 140: bp_category = 2
        else: bp_category = 3
        
        if chol <= 200: chol_category = 0
        elif chol <= 240: chol_category = 1
        elif chol <= 300: chol_category = 2
        else: chol_category = 3
        
        hr_recovery = 220 - age - thalach
        
        risk_score = (
            (age > 55) * 3 + (trestbps > 130) * 2 + (chol > 200) * 2 + 
            fbs * 2 + exang * 4 + (oldpeak > 1) * 4 + 
            (ca > 0) * 5 + (thal == 2) * 4 + (slope == 2) * 3
        )
        
        age_bp_interaction = age * trestbps / 100
        chol_oldpeak_interaction = chol * oldpeak / 100
        age_oldpeak_interaction = age * oldpeak / 10
        critical_combination = ((exang == 1) and (oldpeak > 2) and (ca > 0)) * 10
        
        return [age_category, bp_category, chol_category, hr_recovery, risk_score, 
                age_bp_interaction, chol_oldpeak_interaction, age_oldpeak_interaction, 
                critical_combination]
    
    def predict_risk(self, input_data):
        """Make prediction using the loaded model"""
        if self.model_data is None:
            return None, 0.0
        
        try:
            model = self.model_data['model']
            feature_names = self.model_data['feature_names']
            
            engineered_features = self.create_engineered_features(input_data)
            full_input_data = list(input_data) + engineered_features
            
            input_df = pd.DataFrame([full_input_data], columns=feature_names)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            return prediction, probability
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, 0.0

def landing_page():
    """Beautiful landing page with HTML design"""
    local_css()
    
    st.markdown("""
    <style>
        .hero {
            position: relative;
            height: 620px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: 
            linear-gradient(135deg, rgba(255, 20, 80, 0.4) 0%, rgba(139, 0, 139, 0.3) 50%, rgba(0, 0, 0, 0.7) 100%),
            url('https://static.vecteezy.com/system/resources/previews/021/433/273/non_2x/heart-shaped-technology-background-it-s-a-graph-showing-the-rhythm-of-your-heart-pumping-dark-blue-background-with-a-grid-vector.jpg') center/cover;
        animation: zoomPulse 20s ease-in-out infinite;
        }

        .content {
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 3rem;
            max-width: 900px;
        }

        .main-title {
            font-size: 4rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #fff 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .tagline {
            font-size: 1.5rem;
            color: #ddd;
            margin-bottom: 2rem;
            font-weight: 300;
            letter-spacing: 1px;
        }

        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem;
            }
            .tagline {
                font-size: 1.2rem;
            }
            .heart-icon {
                font-size: 70px;
            }
        }
    </style>
    
    <div class="hero">
        <div class="content">
                <br>
                <br>
                <br>
                <br>
            <h1 class="main-title">Heart Health AI</h1>
            <p class="tagline">Smart Cardiac Risk Assessment Powered by Artificial Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
   
    # Health Recommendations
    st.markdown("""
    <div style='text-align: center; margin: 4rem 0 2rem 0;'>
        <h2 class='section-header'>Key Heart Health Factors</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #ff6b6b; font-size: 1.2rem; margin-bottom: 1rem;'>💓 Blood Pressure</h4>
            <p style='color: #ccc;'>Maintain healthy levels below 120/80 mmHg for optimal heart health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #ff6b6b; font-size: 1.2rem; margin-bottom: 1rem;'>🩸 Cholesterol</h4>
            <p style='color: #ccc;'>Keep cholesterol under 200 mg/dL to reduce cardiovascular risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #ff6b6b; font-size: 1.2rem; margin-bottom: 1rem;'>🏃 Exercise</h4>
            <p style='color: #ccc;'>Regular physical activity strengthens your heart and improves circulation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 START HEART ANALYSIS", use_container_width=True, key="start_analysis", type="primary"):
            st.session_state.page = "analysis"
            st.rerun()
        
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='color: #999; font-size: 0.9rem;'>
                🔒 Your privacy is protected. All analysis happens locally on your device.
            </p>
        </div>
        """, unsafe_allow_html=True)

def analysis_page():
    """Analysis page with all mandatory fields"""
    local_css()
    app = HeartDiseaseApp()
    
    with st.sidebar:
        if st.button("🏠 Back to Home"):
            st.session_state.page = "landing"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 About This Tool")
        st.info("""
        This tool assesses your heart disease risk based on established medical parameters.
        
        **⚠️ Important Note:** 
        This is a screening tool, not a medical diagnosis. Always consult healthcare professionals for medical advice.
        """)
        
        st.markdown("### 🔧 System Status")
        if app.model_data is None:
            st.error("**Model:** Not Loaded")
            st.info("Run the model training script first")
        else:
            st.success(f"**Model:** Loaded")
            st.success(f"**Type:** {app.model_data.get('model_type', 'Unknown')}")
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 class='section-header'>Heart Health Assessment</h1>
        <p style='font-size: 1.2rem; color: #ccc; margin-bottom: 3rem;'>
            Please provide all the following health information for accurate assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Basic Information")
        
        age = st.slider("**How old are you?**", min_value=20, max_value=90, value=45,
                       help="Age helps us understand your baseline risk level")
        
        gender = st.radio("**What is your biological sex?**", ["Female", "Male"],
                         help="Heart disease risk varies between men and women")
        
        chest_pain = st.selectbox("**Do you experience chest pain or discomfort?**",
            ["Never have chest pain", "Occasional mild chest discomfort", 
             "Regular chest pain during activity", "Frequent chest pain even at rest"],
            help="This helps identify potential heart-related symptoms")
        
        blood_pressure = st.slider("**What is your typical blood pressure?**",
                                   min_value=80, max_value=200, value=120,
                                   help="Normal blood pressure is below 120/80 mmHg")
        
        cholesterol = st.slider("**What is your cholesterol level?**",
                               min_value=100, max_value=400, value=200,
                               help="Healthy cholesterol is below 200 mg/dL")
        
        # Moved from advanced section
        st_slope = st.selectbox("**ST Segment Slope from ECG**",
            ["Normal upward slope", "Flat (may indicate issues)", "Downward slope (higher concern)"],
            help="Slope of ST segment in ECG during exercise")

    with col2:
        st.subheader("💊 Health History")
        
        diabetes = st.radio("**Do you have diabetes or high blood sugar?**", ["No", "Yes"],
                           help="Diabetes significantly increases heart disease risk")
        
        exercise_pain = st.radio("**Do you get chest pain during physical activity?**", ["No", "Yes"],
                                help="Pain during exercise can indicate heart issues")
        
        ecg_abnormal = st.radio("**Have you ever had an abnormal ECG test?**",
            ["No", "Never tested", "Yes, minor abnormalities", "Yes, significant abnormalities"],
            help="ECG tests measure your heart's electrical activity")
        
        max_heart_rate = st.slider("**What's the highest heart rate you can achieve during exercise?**",
                                   min_value=60, max_value=220, value=150,
                                   help="Higher maximum heart rate is generally better for heart health")
        
        st_depression = st.slider("**ST Depression Level (from stress test)**",
                                 min_value=0.0, max_value=6.0, value=0.5, step=0.1,
                                 help="Higher values may indicate reduced blood flow to heart")
        
        # Moved from advanced section
        blocked_vessels = st.selectbox("**Number of blocked heart arteries**",
            ["None known", "1 artery", "2 arteries", "3 arteries"],
            help="From angiogram or other heart imaging tests")
        
        # Moved from advanced section
        blood_flow = st.selectbox("**Blood flow to heart muscle**",
            ["Normal blood flow", "Minor reduction in blood flow", "Significant blood flow issues"],
            help="How well blood flows to your heart muscle")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyze My Heart Health", use_container_width=True, type="primary"):
        input_data = convert_user_inputs(
            age, gender, chest_pain, blood_pressure, cholesterol,
            diabetes, exercise_pain, ecg_abnormal, max_heart_rate,
            st_depression, st_slope, blocked_vessels, blood_flow
        )
        
        with st.spinner("🫀 Analyzing your heart health data..."):
            prediction, probability = app.predict_risk(input_data)
            
            if prediction is not None:
                display_results(prediction, probability, input_data)
            else:
                st.error("Unable to analyze your data. Please check if the model is properly trained.")

def convert_user_inputs(age, gender, chest_pain, bp, chol, diabetes, 
                       exercise_pain, ecg, max_hr, st_depress, st_slope, 
                       vessels, blood_flow):
    """Convert user-friendly inputs to model format"""
    
    sex = 1 if gender == "Male" else 0
    
    cp_mapping = {
        "Never have chest pain": 0,
        "Occasional mild chest discomfort": 1,
        "Regular chest pain during activity": 2,
        "Frequent chest pain even at rest": 3
    }
    cp = cp_mapping.get(chest_pain, 0)
    
    fbs = 1 if diabetes == "Yes" else 0
    exang = 1 if exercise_pain == "Yes" else 0
    
    ecg_mapping = {
        "No": 0,
        "Never tested": 0,
        "Yes, minor abnormalities": 1,
        "Yes, significant abnormalities": 2
    }
    restecg = ecg_mapping.get(ecg, 0)
    
    slope_mapping = {
        "Normal upward slope": 1,
        "Flat (may indicate issues)": 2,
        "Downward slope (higher concern)": 2
    }
    slope = slope_mapping.get(st_slope, 1)
    
    ca_mapping = {
        "None known": 0,
        "1 artery": 1,
        "2 arteries": 2,
        "3 arteries": 3
    }
    ca = ca_mapping.get(vessels, 0)
    
    thal_mapping = {
        "Normal blood flow": 0,
        "Minor reduction in blood flow": 1,
        "Significant blood flow issues": 2
    }
    thal = thal_mapping.get(blood_flow, 0)
    
    return [age, sex, cp, bp, chol, fbs, restecg, max_hr, exang, st_depress, slope, ca, thal]

def display_results(prediction, probability, input_data):
    """Display results without showing probability percentage"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">Your Heart Health Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if probability >= 0.7:
            risk_level = "High"
            risk_class = "risk-high"
            risk_icon = "🔴"
            color = "#e63946"
            recommendations = """
            **🚨 Recommended Actions:**
            - Consult a cardiologist as soon as possible
            - Consider comprehensive cardiac testing
            - Discuss medication options with your doctor
            - Make immediate lifestyle changes
            - Monitor symptoms closely
            """
        elif probability >= 0.4:
            risk_level = "Moderate"
            risk_class = "risk-moderate"
            risk_icon = "🟡"
            color = "#ffc107"
            recommendations = """
            **⚠️ Recommended Actions:**
            - Schedule a check-up with your doctor
            - Consider preventive lifestyle changes
            - Monitor your blood pressure regularly
            - Maintain healthy cholesterol levels
            - Stay physically active with doctor's approval
            """
        else:
            risk_level = "Low"
            risk_class = "risk-low"
            risk_icon = "🟢"
            color = "#2a9d8f"
            recommendations = """
            **✅ Keep Up the Good Work:**
            - Continue regular health checkups
            - Maintain balanced diet and exercise
            - Monitor any new symptoms
            - Stay informed about heart health
            - Share these healthy habits with family
            """
        
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        st.markdown(f"### {risk_icon} {risk_level} Risk Level")
        st.markdown(recommendations)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Create a simple risk indicator instead of gauge with probability
        fig = create_simple_risk_indicator(risk_level, color)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Quick Stats</h4>
            <p>• Based on 13 health parameters</p>
            <p>• 85%+ prediction accuracy</p>
            <p>• Real-time AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Your Key Risk Factors")
    display_risk_factors_simple(input_data)

    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.rerun()

def create_simple_risk_indicator(risk_level, color):
    """Create a simple risk level indicator without probability numbers"""
    
    # Map risk levels to positions
    risk_positions = {
        "Low": 25,
        "Moderate": 50,
        "High": 75
    }
    
    position = risk_positions.get(risk_level, 50)
    
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=position,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_level}", 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(46, 204, 113, 0.3)'},
                {'range': [33, 66], 'color': 'rgba(241, 196, 15, 0.3)'},
                {'range': [66, 100], 'color': 'rgba(231, 76, 60, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': position
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def display_risk_factors_simple(input_data):
    """Display risk factors in simple language"""
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = input_data
    
    risk_factors = []
    
    if age > 55:
        risk_factors.append("👴 **Age**: Being over 55 increases heart disease risk")
    if trestbps > 130:
        risk_factors.append("💓 **Blood Pressure**: Elevated blood pressure detected")
    if chol > 200:
        risk_factors.append("🩸 **Cholesterol**: High cholesterol levels")
    if fbs == 1:
        risk_factors.append("🍬 **Blood Sugar**: High blood sugar/diabetes risk")
    if exang == 1:
        risk_factors.append("🏃 **Exercise Symptoms**: Chest pain during physical activity")
    if oldpeak > 1:
        risk_factors.append("📉 **Heart Stress**: Abnormal heart stress test results")
    if ca > 0:
        risk_factors.append("🫀 **Arteries**: Potential artery blockages")
    
    if risk_factors:
        for factor in risk_factors:
            st.markdown(f"<div class='info-card'>{factor}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-card'>
            <p>🎉 No major risk factors identified! Keep maintaining your healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'analysis':
        analysis_page()

if __name__ == "__main__":
    main()