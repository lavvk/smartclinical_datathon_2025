"""
Streamlit frontend for SmartClinical
Hospital Resource Allocation Dashboard
"""
import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SmartClinical - Resource Allocation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #ffd93d;
        color: black;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .risk-normal {
        background-color: #6bcf7f;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_single_patient(vitals):
    """Predict risk for a single patient"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"patient": vitals},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def allocate_resources(patients, oxygen, staff):
    """Allocate resources for a batch of patients"""
    try:
        response = requests.post(
            f"{API_URL}/allocate",
            json={
                "patients": patients,
                "available_oxygen": oxygen,
                "available_staff": staff
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_risk_color(risk_level):
    """Get color class for risk level"""
    colors = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
        "Normal": "risk-normal"
    }
    return colors.get(risk_level, "")

def main():
    # Header
    st.markdown('<div class="main-header">🏥 SmartClinical</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Risk-Driven Hospital Resource Allocation System</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the API server first:")
        st.code("python api.py", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        
        # Resource inputs
        st.subheader("Available Resources")
        available_oxygen = st.number_input(
            "Oxygen Units Available",
            min_value=0,
            value=10,
            step=1,
            help="Number of oxygen units available for allocation"
        )
        available_staff = st.number_input(
            "Staff Members Available",
            min_value=0,
            value=5,
            step=1,
            help="Number of staff members available for allocation"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a CSV file with patient vitals OR enter manually
        2. Click "Run Allocation" to get resource recommendations
        3. Review the allocation results
        4. Export the report if needed
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This is an advisory tool for resource planning.
        Not a diagnostic or clinical decision-making system.
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Batch Allocation", "👤 Single Patient", "ℹ️ About"])
    
    with tab1:
        st.header("Batch Patient Allocation")
        
        # Data input method
        input_method = st.radio(
            "Select input method:",
            ["Upload CSV", "Manual Entry"],
            horizontal=True
        )
        
        patients_data = []
        
        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload patient data CSV",
                type=["csv"],
                help="CSV should have columns: Respiratory_Rate, Oxygen_Saturation, O2_Scale, Systolic_BP, Heart_Rate, Temperature, Consciousness, On_Oxygen"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} patients from CSV")
                    
                    # Display preview
                    with st.expander("Preview Data"):
                        st.dataframe(df.head(10))
                    
                    # Convert to API format
                    required_cols = ["Respiratory_Rate", "Oxygen_Saturation", "O2_Scale", 
                                   "Systolic_BP", "Heart_Rate", "Temperature", "Consciousness", "On_Oxygen"]
                    
                    if all(col in df.columns for col in required_cols):
                        for idx, row in df.iterrows():
                            patient_id = row.get("Patient_ID", f"P{idx+1:04d}")
                            patients_data.append({
                                "patient_id": str(patient_id),
                                "Respiratory_Rate": float(row["Respiratory_Rate"]),
                                "Oxygen_Saturation": float(row["Oxygen_Saturation"]),
                                "O2_Scale": int(row["O2_Scale"]),
                                "Systolic_BP": float(row["Systolic_BP"]),
                                "Heart_Rate": float(row["Heart_Rate"]),
                                "Temperature": float(row["Temperature"]),
                                "Consciousness": str(row["Consciousness"]),
                                "On_Oxygen": int(row["On_Oxygen"])
                            })
                    else:
                        st.error(f"Missing required columns. Need: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        else:  # Manual Entry
            st.subheader("Enter Patient Vitals")
            num_patients = st.number_input("Number of patients", min_value=1, max_value=50, value=3)
            
            for i in range(num_patients):
                with st.expander(f"Patient {i+1}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        patient_id = st.text_input(f"Patient ID {i+1}", value=f"P{i+1:04d}", key=f"id_{i}")
                        respiratory_rate = st.number_input(f"Respiratory Rate", min_value=0.0, max_value=60.0, value=20.0, step=0.1, key=f"rr_{i}")
                        oxygen_saturation = st.number_input(f"Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1, key=f"o2_{i}")
                        o2_scale = st.selectbox(f"O2 Scale", [1, 2, 3], key=f"o2s_{i}")
                        systolic_bp = st.number_input(f"Systolic BP", min_value=0.0, max_value=300.0, value=120.0, step=0.1, key=f"bp_{i}")
                    
                    with col2:
                        heart_rate = st.number_input(f"Heart Rate", min_value=0.0, max_value=300.0, value=80.0, step=0.1, key=f"hr_{i}")
                        temperature = st.number_input(f"Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, key=f"temp_{i}")
                        consciousness = st.selectbox(f"Consciousness", ["A", "P", "U", "V"], key=f"cons_{i}")
                        on_oxygen = st.selectbox(f"On Oxygen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"oxy_{i}")
                    
                    patients_data.append({
                        "patient_id": patient_id,
                        "Respiratory_Rate": respiratory_rate,
                        "Oxygen_Saturation": oxygen_saturation,
                        "O2_Scale": o2_scale,
                        "Systolic_BP": systolic_bp,
                        "Heart_Rate": heart_rate,
                        "Temperature": temperature,
                        "Consciousness": consciousness,
                        "On_Oxygen": on_oxygen
                    })
        
        # Run allocation button
        if patients_data:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_allocation = st.button("🚀 Run Allocation", type="primary", use_container_width=True)
            
            if run_allocation:
                with st.spinner("Processing allocation..."):
                    result = allocate_resources(patients_data, available_oxygen, available_staff)
                    
                    if result:
                        st.success("✅ Allocation complete!")
                        
                        # Display summary
                        summary = result["summary"]
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("Total Patients", summary["total_patients"])
                        with col2:
                            st.metric("High Risk", summary["high_risk"], delta=None)
                        with col3:
                            st.metric("Medium Risk", summary["medium_risk"])
                        with col4:
                            st.metric("Oxygen Allocated", summary["oxygen_allocated"], 
                                    delta=f"{summary['available_oxygen'] - summary['oxygen_allocated']} remaining")
                        with col5:
                            st.metric("Staff Allocated", summary["staff_allocated"],
                                    delta=f"{summary['available_staff'] - summary['staff_allocated']} remaining")
                        with col6:
                            st.metric("Low/Normal", summary["low_risk"] + summary["normal"])
                        
                        # Allocation table
                        st.markdown("### 📋 Allocation Results")
                        
                        # Prepare dataframe
                        allocation_df = []
                        for patient in result["allocation"]:
                            allocation_df.append({
                                "Patient ID": patient["patient_id"],
                                "Risk Level": patient["risk_level"],
                                "Risk Score": f"{patient['risk_score']:.3f}",
                                "Oxygen": "✅" if patient["allocated_oxygen"] else "❌",
                                "Staff": "✅" if patient["allocated_staff"] else "❌",
                                "Resp Rate": patient["vitals"]["Respiratory_Rate"],
                                "O2 Sat": patient["vitals"]["Oxygen_Saturation"],
                                "BP": patient["vitals"]["Systolic_BP"],
                                "Heart Rate": patient["vitals"]["Heart_Rate"],
                                "Temp": patient["vitals"]["Temperature"]
                            })
                        
                        df_display = pd.DataFrame(allocation_df)
                        
                        # Color code by risk level
                        def color_risk(val):
                            if val == "High":
                                return "background-color: #ff6b6b; color: white"
                            elif val == "Medium":
                                return "background-color: #ffa500; color: white"
                            elif val == "Low":
                                return "background-color: #ffd93d"
                            elif val == "Normal":
                                return "background-color: #6bcf7f; color: white"
                            return ""
                        
                        styled_df = df_display.style.applymap(color_risk, subset=["Risk Level"])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution pie chart
                            risk_counts = {
                                "High": summary["high_risk"],
                                "Medium": summary["medium_risk"],
                                "Low": summary["low_risk"],
                                "Normal": summary["normal"]
                            }
                            fig_pie = px.pie(
                                values=list(risk_counts.values()),
                                names=list(risk_counts.keys()),
                                title="Risk Level Distribution",
                                color_discrete_map={
                                    "High": "#ff6b6b",
                                    "Medium": "#ffa500",
                                    "Low": "#ffd93d",
                                    "Normal": "#6bcf7f"
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Resource allocation bar chart
                            fig_bar = go.Figure()
                            fig_bar.add_trace(go.Bar(
                                name="Available",
                                x=["Oxygen", "Staff"],
                                y=[summary["available_oxygen"], summary["available_staff"]],
                                marker_color="lightblue"
                            ))
                            fig_bar.add_trace(go.Bar(
                                name="Allocated",
                                x=["Oxygen", "Staff"],
                                y=[summary["oxygen_allocated"], summary["staff_allocated"]],
                                marker_color="darkblue"
                            ))
                            fig_bar.update_layout(
                                title="Resource Allocation",
                                barmode="group",
                                yaxis_title="Count"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Export options
                        st.markdown("### 📥 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV export
                            csv_data = []
                            for patient in result["allocation"]:
                                csv_data.append({
                                    "Patient_ID": patient["patient_id"],
                                    "Risk_Level": patient["risk_level"],
                                    "Risk_Score": patient["risk_score"],
                                    "Oxygen_Allocated": patient["allocated_oxygen"],
                                    "Staff_Allocated": patient["allocated_staff"],
                                    **patient["vitals"]
                                })
                            csv_df = pd.DataFrame(csv_data)
                            csv_str = csv_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv_str,
                                file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON export
                            json_str = json.dumps(result, indent=2)
                            st.download_button(
                                label="📋 Download JSON",
                                data=json_str,
                                file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
    
    with tab2:
        st.header("Single Patient Risk Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Vitals")
            respiratory_rate = st.number_input("Respiratory Rate", min_value=0.0, max_value=60.0, value=20.0, step=0.1, key="single_rr")
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1, key="single_o2")
            o2_scale = st.selectbox("O2 Scale", [1, 2, 3], key="single_o2s")
            systolic_bp = st.number_input("Systolic BP", min_value=0.0, max_value=300.0, value=120.0, step=0.1, key="single_bp")
        
        with col2:
            st.subheader("")
            heart_rate = st.number_input("Heart Rate", min_value=0.0, max_value=300.0, value=80.0, step=0.1, key="single_hr")
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, key="single_temp")
            consciousness = st.selectbox("Consciousness", ["A", "P", "U", "V"], key="single_cons")
            on_oxygen = st.selectbox("On Oxygen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="single_oxy")
        
        if st.button("🔍 Predict Risk", type="primary"):
            vitals = {
                "Respiratory_Rate": respiratory_rate,
                "Oxygen_Saturation": oxygen_saturation,
                "O2_Scale": o2_scale,
                "Systolic_BP": systolic_bp,
                "Heart_Rate": heart_rate,
                "Temperature": temperature,
                "Consciousness": consciousness,
                "On_Oxygen": on_oxygen
            }
            
            with st.spinner("Predicting risk..."):
                result = predict_single_patient(vitals)
                
                if result:
                    st.success("✅ Prediction complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_level = result["risk_level"]
                        risk_score = result["risk_score"]
                        
                        st.markdown(f"### Risk Level: <span class='{get_risk_color(risk_level)}'>{risk_level}</span>", unsafe_allow_html=True)
                        st.metric("Risk Score", f"{risk_score:.3f}")
                    
                    with col2:
                        st.markdown("### Probability Distribution")
                        prob_df = pd.DataFrame({
                            "Risk Level": list(result["probabilities"].keys()),
                            "Probability": list(result["probabilities"].values())
                        })
                        fig = px.bar(
                            prob_df,
                            x="Risk Level",
                            y="Probability",
                            color="Risk Level",
                            color_discrete_map={
                                "High": "#ff6b6b",
                                "Medium": "#ffa500",
                                "Low": "#ffd93d",
                                "Normal": "#6bcf7f"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("About SmartClinical")
        st.markdown("""
        ### Overview
        SmartClinical is a web-based clinical decision-support tool that predicts patient risk levels 
        from vital signs and automatically recommends optimized allocation of hospital resources.
        
        ### Features
        - **Risk Prediction**: ML-powered risk assessment using patient vitals
        - **Resource Allocation**: Automated allocation of oxygen units and staff based on risk
        - **Batch Processing**: Handle multiple patients simultaneously
        - **Export Reports**: Download allocation results as CSV or JSON
        
        ### How It Works
        1. Upload patient vital signs data (CSV or manual entry)
        2. Specify available resources (oxygen units, staff)
        3. System predicts risk levels and allocates resources optimally
        4. Review and export allocation recommendations
        
        ### Technical Details
        - **Backend**: FastAPI with RandomForest classifier
        - **Frontend**: Streamlit dashboard
        - **Model**: Trained on historical patient data with risk labels
        
        ### ⚠️ Important Disclaimer
        This system is **advisory only** and is not a diagnostic tool. It is designed for 
        workflow support and resource planning. All clinical decisions should be made by 
        qualified healthcare professionals.
        """)

if __name__ == "__main__":
    main()

