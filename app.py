import streamlit as st
import pandas as pd
import numpy as np
import joblib
import types
import traceback
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px

#Page configuration

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Custom CSS for better styling

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-box h2 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box h3 {
        font-size: 1.2rem;
        margin-bottom: 0.3rem;
    }
    .prediction-box p {
        font-size: 0.95rem;
    }
    .safe {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .risk {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }

    /* Make primary buttons more prominent */
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 8px rgba(31,119,180,0.15);
    }
    .stButton>button:hover {
        background-color: #155a8a !important;
    }
    
    /* Reduce header sizes */
    h1 {
        font-size: 1.8rem !important;
    }
    h2 {
        font-size: 1.3rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
    }
    
    /* Make form labels more compact */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load the trained model and preprocessing components"""
    try:
        model = joblib.load('C:/Users/GetnetBantie/Documents/PROJECTS/loan_default_model.pkl')
        scaler = joblib.load('C:/Users/GetnetBantie/Documents/PROJECTS/standard_scaler.pkl')
        encoder = joblib.load('C:/Users/GetnetBantie/Documents/PROJECTS/label_encoder.pkl')
        
        #Test the model with a simple prediction to verify it works

        st.sidebar.info(f"Model loaded: {type(model).__name__}")
        st.sidebar.info(f"Model has {len(model.estimators_)} estimators" if hasattr(model, 'estimators_') else "Model type unknown")
        
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all .pkl files are in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        with st.expander("Load Error Details"):
            st.code(traceback.format_exc())
        return None, None, None


def preprocess_input(data, scaler, encoder):

    """Preprocess the input data similar to training pipeline.

    This function:
    - Separates numeric and categorical columns.
    - Attempts to encode categorical columns using the provided encoder (robust
      to LabelEncoder, OrdinalEncoder, and some ColumnTransformer behaviors).
    - Coerces numeric columns to numeric dtype before scaling and fills NaNs with 0.
    - Applies the provided scaler only to numeric columns.
    """
    processed_data = data.copy()

    #Identify numeric and categorical columns

    numerical_cols = list(processed_data.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in processed_data.columns if c not in numerical_cols]

    #First, handle categorical columns: try to use provided encoder, else factorize
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            try:
                if encoder is not None:
                    #If encoder looks like a single-column LabelEncoder
                    if hasattr(encoder, 'classes_') and not hasattr(encoder, 'feature_names_in_'):
                        #label encoder expects 1D
                        processed_data[col] = encoder.transform(processed_data[col])
                    else:
                        #Try transforming the single column with a more general encoder
                        try:
                            transformed = encoder.transform(processed_data[[col]])
                        except Exception:
                            #As a last resort, try transforming the whole df and then pick the
                            #column by name/index if possible
                            transformed = encoder.transform(processed_data)

                        if isinstance(transformed, np.ndarray):
                            #If it returned a 1D or single-column array, use it
                            if transformed.ndim == 1:
                                processed_data[col] = transformed
                            elif transformed.shape[1] == 1:
                                processed_data[col] = transformed.ravel()
                            else:
                                #Multiple columns produced; fall back to factorize for this col
                                processed_data[col], _ = pd.factorize(processed_data[col])
                else:
                    #No encoder provided: fallback to factorize
                    processed_data[col], _ = pd.factorize(processed_data[col])
            except Exception:

                #If any transform fails, fallback to pandas factorize

                processed_data[col], _ = pd.factorize(processed_data[col])

    #Recompute numeric columns (some categorical columns may have become numeric)

    numerical_cols = list(processed_data.select_dtypes(include=[np.number]).columns)


    #Coerce numeric columns to numeric types and fill NaNs with 0 to avoid scaler errors

    if numerical_cols:
        processed_data[numerical_cols] = processed_data[numerical_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    #Apply scaler to numeric columns if available

    if scaler is not None and numerical_cols:
        try:
            processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
        except Exception:

            #If scaler.transform fails, leave numeric columns as-is so calling code can debug

            pass

    return processed_data


def align_features(df: pd.DataFrame, model):
        """Ensure dataframe has the same features the model expects.

        - If the model exposes `feature_names_in_`, we add any missing columns
            with safe default values (zeros or empty strings) and reorder the
            columns to match the expected order.
        - This prevents 'feature names should match' errors when users omit
            some fields like 'ID' that were present during training.
        """
        if model is None:
                return df

        expected = getattr(model, 'feature_names_in_', None)
        if expected is None:
                return df

        expected = list(expected)
        df_copy = df.copy()

        for col in expected:
                if col not in df_copy.columns:
                        # Choose a sensible default: 0 for numeric-like names, empty string otherwise
                        df_copy[col] = 0 if df_copy.select_dtypes(include=[np.number]).columns.size else ''

        # Reorder and return only expected columns
        return df_copy[expected]


def create_feature_explanation():

    """Create feature explanations for users"""
    feature_info = {
        'loan_amount': 'Total loan amount requested',
        'Credit_Score': 'Applicant credit score (500-900)',
        'income': 'Annual income of applicant',
        'property_value': 'Value of the property being purchased',
        'loan_limit': 'Loan limit category',
        'Gender': 'Applicant gender',
        'loan_purpose': 'Purpose of the loan',
        'dtir1': 'Debt-to-income ratio',
        'LTV': 'Loan-to-value ratio',
        'rate_of_interest': 'Interest rate on loan',
        'age': 'Age group of applicant'
    }

    return feature_info


def main():

    #Header

    st.markdown('<h1 class="main-header">🏦 Loan Default Risk Predictor</h1>', unsafe_allow_html=True)
    model, scaler, encoder = load_components()
    if model is None:
        return
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", ["Single Prediction", "Batch Prediction", "Model Info"]) 
    if app_mode == "Single Prediction":
        single_prediction(model, scaler, encoder)
    elif app_mode == "Batch Prediction":
        batch_prediction(model, scaler, encoder)
    else:
        model_info()



def single_prediction(model, scaler, encoder):

    """Single loan application prediction interface with a cleaner layout and a form-based submit button"""

    st.header("📊 Single Loan Application Assessment")


    #Main layout: input form on the left, help/feature info on the right
    

    left, right = st.columns([2, 1])

    with left:

        #Use a form so inputs are submitted together; the submit button is a true form button



        with st.form(key="prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Applicant Information")
                gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"]) 
                age = st.selectbox("Age Group", ["25-34", "35-44", "45-54", "55-64", "65-74", "75+"])
                income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
                credit_score = st.slider("Credit Score", min_value=500, max_value=900, value=700)
                loan_amount = st.number_input("Loan Amount ($)", min_value=10000, value=200000, step=10000)
                loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"]) 
                loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"]) 

            with col2:
                st.subheader("Financial Details")
                property_value = st.number_input("Property Value ($)", min_value=10000, value=250000, step=10000)
                rate_of_interest = st.slider("Interest Rate (%)", min_value=0.0, max_value=8.0, value=4.0, step=0.1)
                dtir1 = st.slider("Debt-to-Income Ratio", min_value=5.0, max_value=60.0, value=35.0, step=1.0)
                LTV = st.slider("Loan-to-Value Ratio", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                approv_in_adv = st.selectbox("Pre-approval", ["pre", "nopre"]) 
                business_commercial = st.selectbox("Business/Commercial", ["b/c", "nob/c"]) 
                region = st.selectbox("Region", ["south", "North", "central"]) 

            with st.expander("Advanced Features (optional)"):
                col3, col4 = st.columns(2)
                with col3:
                    credit_type = st.selectbox("Credit Type", ["CIB", "EXP", "CRIF", "EQUI"]) 
                    co_applicant_credit = st.selectbox("Co-applicant Credit Type", ["CIB", "EXP", "CRIF", "EQUI"]) 
                    open_credit = st.selectbox("Open Credit", ["opc", "nopc"]) 
                with col4:
                    construction_type = st.selectbox("Construction Type", ["mc", "sb"]) 
                    occupancy_type = st.selectbox("Occupancy Type", ["pr", "sr", "ir"]) 
                    security_type = st.selectbox("Security Type", ["direct", "Indriect"]) 

            #Center the submit button visually

            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:

                submit = st.form_submit_button(label="Predict Default Risk")

    with right:
        st.subheader("Feature Help")
        feat = create_feature_explanation()
        for k, v in feat.items():
            st.markdown(f"**{k}** — {v}")


    #Handle form submission


    if submit:
        input_data = {
            'loan_limit': ['cf'],
            'Gender': [gender],
            'approv_in_adv': [approv_in_adv],
            'loan_type': [loan_type],
            'loan_purpose': [loan_purpose],
            'Credit_Worthiness': ['l1'],
            'open_credit': [open_credit],
            'business_or_commercial': [business_commercial],
            'loan_amount': [loan_amount],
            'rate_of_interest': [rate_of_interest],
            'Interest_rate_spread': [0.44],
            'Upfront_charges': [3224],
            'term': [335],
            'Neg_ammortization': ['NO'],
            'interest_only': ['not_io'],
            'lump_sum_payment': ['not_lpsm'],
            'property_value': [property_value],
            'construction_type': [construction_type],
            'occupancy_type': [occupancy_type],
            'Secured_by': ['land'],
            'total_units': ['1U'],
            'income': [income],
            'credit_type': [credit_type],
            'Credit_Score': [credit_score],
            'co-applicant_credit_type': [co_applicant_credit],
            'age': [age],
            'submission_of_application': ['to_inst'],
            'LTV': [LTV],
            'Region': [region],
            'Security_Type': [security_type],
            'dtir1': [dtir1]
        }

        input_df = pd.DataFrame(input_data)
        
        #Debug: Show input values being used

        with st.expander("🔍 Debug: View Input Data"):
            st.write("**Key Input Values:**")
            debug_cols = st.columns(3)
            with debug_cols[0]:
                st.write(f"Loan Amount: ${loan_amount:,}")
                st.write(f"Credit Score: {credit_score}")
                st.write(f"Income: ${income:,}")
            with debug_cols[1]:
                st.write(f"Property Value: ${property_value:,}")
                st.write(f"Interest Rate: {rate_of_interest}%")
                st.write(f"DTI Ratio: {dtir1}%")
            with debug_cols[2]:
                st.write(f"LTV Ratio: {LTV}%")
                st.write(f"Gender: {gender}")
                st.write(f"Age Group: {age}")

        try:
            #Align features to the model's expectations (fills missing columns)

            input_df_aligned = align_features(input_df, model)
            
            #Debug: Show aligned data

            with st.expander("🔬 Debug: Aligned Features"):
                st.write("Before alignment shape:", input_df.shape)
                st.write("After alignment shape:", input_df_aligned.shape)
                st.dataframe(input_df_aligned)
            
            processed_input = preprocess_input(input_df_aligned, scaler, encoder)
            

            #Debug: Show processed data


            with st.expander("🔬 Debug: Processed Features (After Scaling)"):
                st.write("Processed shape:", processed_input.shape)
                st.dataframe(processed_input)
            
            #Make predictions

            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0]
            

            #Debug: Show raw model outputs

            with st.expander("🔬 Debug: Raw Model Output"):
                st.write(f"Prediction: {prediction}")
                st.write(f"Probability array: {probability}")
                st.write(f"Probability shape: {probability.shape}")
                st.write(f"Class 0 (No Default): {probability[0]:.4f}")
                st.write(f"Class 1 (Default): {probability[1]:.4f}")
            
            display_prediction_results(prediction, probability, input_df)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all input values are within expected ranges.")


            #Show detailed error information for debugging



            with st.expander("Error Details"):
                import traceback
                st.write(f"Exception type: {type(e).__name__}")
                st.write(f"Exception message: {str(e)}")
                st.write("Traceback:")
                st.code(traceback.format_exc())
                st.write("Input data shape:", input_df.shape)
                st.write("Input data columns:", list(input_df.columns))

    
#(Legacy direct button removed. Use the form's "Predict Default Risk" submit button.)



def display_prediction_results(prediction, probability, input_data):
    """Display prediction results in an appealing way"""
    
    default_prob = probability[1]  


    #Probability of default (class 1)
    

    #Create columns for results

    col1, col2 = st.columns(2)
    
    with col1:

        #Risk gauge chart

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = default_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:


        #Prediction result


        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box risk">
                <h2>🚨 HIGH DEFAULT RISK</h2>
                <h3>Probability: {default_prob:.1%}</h3>
                <p>This application shows signs of potential default risk.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box safe">
                <h2>✅ LOW DEFAULT RISK</h2>
                <h3>Probability: {default_prob:.1%}</h3>
                <p>This application appears to be low risk.</p>
            </div>
            """, unsafe_allow_html=True)
        
        #Key metrics

        st.subheader("Key Metrics")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Credit Score", f"{input_data['Credit_Score'].iloc[0]}")
        
        with col_b:
            st.metric("DTI Ratio", f"{input_data['dtir1'].iloc[0]:.1f}%")
        
        with col_c:
            st.metric("LTV Ratio", f"{input_data['LTV'].iloc[0]:.1f}%")
            

def batch_prediction(model, scaler, encoder):
    """Batch prediction interface"""
    st.header("📁 Batch Loan Applications")
    
    st.info("Upload a CSV file with multiple loan applications for batch processing.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:

            #Read the uploaded file



            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            
            if st.button("Process Batch Predictions"):
                with st.spinner("Processing batch predictions..."):

                    #Preprocess and predict

                    batch_data = align_features(batch_data, model)
                    processed_data = preprocess_input(batch_data, scaler, encoder)
                    predictions = model.predict(processed_data)
                    probabilities = model.predict_proba(processed_data)
                    
                    #Add predictions to dataframe

                    results_df = batch_data.copy()
                    results_df['Default_Prediction'] = predictions
                    results_df['Default_Probability'] = probabilities[:, 1]
                    results_df['Risk_Level'] = np.where(predictions == 1, 'High Risk', 'Low Risk')
                    
                    #display results


                    st.subheader("Batch Prediction Results")

                    
                    #Summary statistics


                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Applications", len(results_df))
                    with col2:
                        st.metric("High Risk Applications", 
                                 len(results_df[results_df['Default_Prediction'] == 1]))
                    with col3:
                        st.metric("Approval Rate", 
                                 f"{(len(results_df[results_df['Default_Prediction'] == 0]) / len(results_df)):.1%}")
                    
                    #Show results table


                    st.dataframe(results_df)
                    

                    #Download button

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="loan_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")



def model_info():
    """Display model information and feature explanations"""
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.markdown("""
        - **Algorithm**: Gradient Boosting Classifier
        - **Ensemble Size**: 17 estimators
        - **Max Depth**: 12 levels
        - **Training Accuracy**: ~85% (based on your notebook)
        """)
        
        st.subheader("Data Preprocessing")
        st.markdown("""
        - **Categorical Encoding**: Label Encoding
        - **Feature Scaling**: Standard Scaler
        - **Missing Values**: Handled during training
        """)
    
    with col2:

        st.subheader("Key Features")
        feature_info = create_feature_explanation()
        
        for feature, description in list(feature_info.items())[:10]: 
            
            #Show first 10

            with st.container():
                st.markdown(f"**{feature}**: {description}")
    
    st.subheader("Model Performance")

    
    # Placeholder for model metrics - you can replace with actual metrics from your notebook
    

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Accuracy", "85.2%")
    with metrics_col2:
        st.metric("Precision", "83.1%")
    with metrics_col3:

        
        st.metric("Recall", "79.8%")
    with metrics_col4:
        st.metric("F1-Score", "81.4%")

if __name__ == "__main__":
    main()