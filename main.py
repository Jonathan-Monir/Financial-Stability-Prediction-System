
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Stability Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c5aa0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-container {
        background-color: #1e1e1e; /* dark background */
        color: white;              /* white text */
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50; /* accent color */
    }
    .metric-container h2,
    .metric-container h3,
    .metric-container p {
        color: white;
        margin: 0;
    }
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2c5aa0;
    }
    .equation-box {
        background-color: #1f4e79;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #d1d3d4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_xgboost_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found at 'models/xgb_best.joblib'. Please ensure the model file exists.")
        return None

# Define column information
COLUMNS = ['year', 'hhis', 'hhig', 'hhic', 'hhit', 'ccr', 'mcr', 'asset_size', 
          'ownership', 'covid', 'inflation', 'bank_age']

COLUMN_DESCRIPTIONS = {
    'year': 'Year of observation',
    'hhis': 'Herfindahl-Hirschman Index (Share)',
    'hhig': 'Herfindahl-Hirschman Index (Geography)', 
    'hhic': 'Herfindahl-Hirschman Index (Concentration)',
    'hhit': 'Herfindahl-Hirschman Index (Total)',
    'ccr': 'Capital Conservation Ratio',
    'mcr': 'Minimum Capital Ratio',
    'asset_size': 'Bank Asset Size',
    'ownership': 'Ownership Structure (0/1)',
    'covid': 'COVID-19 Period (0/1)',
    'inflation': 'Inflation Rate',
    'bank_age': 'Bank Age (years)'
}

def main():
    st.markdown('<h1 class="main-header">Financial Stability Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["🔮 Prediction", "📊 Model Performance"])
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    if page == "🔮 Prediction":
        prediction_page(model)
    elif page == "📈 Linear Regression Analysis":
        linear_regression_page()
    elif page == "📊 Model Performance":
        performance_page(model)

def prediction_page(model):
    st.markdown('<h2 class="sub-header">Financial Stability Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Input Method")
        input_method = st.radio("Choose input method:", 
                               ["Upload CSV File", "Manual Input"])
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Check if required columns exist
                missing_cols = [col for col in COLUMNS if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    return

                # Prepare features for prediction (use original columns in correct order)
                X = df[COLUMNS]

                # Make predictions
                if st.button("Generate Predictions", type="primary"):
                    try:
                        raw_preds = model.predict(X)
                    except Exception as e:
                        st.error(f"Model prediction failed: {e}")
                        raise

                    # normalize prediction shape
                    preds = np.asarray(raw_preds).ravel()
                    df['fs_predicted'] = preds

                    st.subheader("Prediction Results")
                    st.dataframe(df, use_container_width=True)

                    # If true labels present, compute metrics
                    if 'fs' in df.columns:
                        y_true = np.asarray(df['fs']).ravel()

                        # align lengths
                        if len(y_true) != len(preds):
                            minlen = min(len(y_true), len(preds))
                            y_true = y_true[:minlen]
                            preds = preds[:minlen]
                            st.warning(f"Trimmed to first {minlen} rows to match true/predicted lengths.")

                        if len(y_true) < 3:
                            st.info("Accuracy/metrics not calculated for datasets with less than 3 rows.")
                        else:
                            # Heuristic: classification if integer-like or few unique labels
                            unique_true = np.unique(y_true)
                            unique_pred = np.unique(preds)
                            n_unique = max(len(unique_true), len(unique_pred))
                            is_integer_like = np.all(np.mod(y_true, 1) == 0) and np.all(np.mod(preds, 1) == 0)
                            is_classification = is_integer_like or (n_unique <= 10)

                            if is_classification:
                                # build consistent integer encoding for labels
                                labels = np.unique(np.concatenate([unique_true, unique_pred]))
                                y_true_int = pd.Categorical(y_true, categories=labels).codes
                                y_pred_int = pd.Categorical(preds, categories=labels).codes

                                accuracy = accuracy_score(y_true_int, y_pred_int)
                                st.markdown(
                                    f'<div class="metric-container"><h3>Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)</h3></div>',
                                    unsafe_allow_html=True
                                )

                                cm = confusion_matrix(y_true_int, y_pred_int)
                                fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
                                st.plotly_chart(fig_cm, use_container_width=True)

                                report = classification_report(y_true_int, y_pred_int, output_dict=True, zero_division=0)
                                report_df = pd.DataFrame(report).transpose()
                                # map integer label indices back to original label names if possible
                                if len(labels) == report_df.shape[0] or report_df.index.dtype == int:
                                    name_map = {i: str(labels[i]) for i in range(len(labels))}
                                    report_df.index = [name_map.get(idx, idx) for idx in report_df.index]
                                st.subheader("Classification Report")
                                st.dataframe(report_df, use_container_width=True)

                            else:
                                # Regression metrics
                                mse = mean_squared_error(y_true, preds)
                                mae = mean_absolute_error(y_true, preds)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_true, preds)

                                st.markdown(
                                    f"""
                                    <div style="background-color:#1e1e1e; padding:12px; border-radius:8px;">
                                        <h3 style="color:white; margin:0;">Regression Metrics</h3>
                                        <p style="color:white; font-size:16px; margin:4px 0 0 0;">
                                            MSE: {mse:.4f} &nbsp;&nbsp;
                                            RMSE: {rmse:.4f} &nbsp;&nbsp;
                                            MAE: {mae:.4f} &nbsp;&nbsp;
                                            R²: {r2:.4f}
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # True vs Predicted scatter with identity line
                                fig_scatter = px.scatter(
                                    x=y_true, y=preds,
                                    labels={'x': 'True', 'y': 'Predicted'},
                                    title='True vs Predicted'
                                )
                                minv = float(np.min([np.min(y_true), np.min(preds)]))
                                maxv = float(np.max([np.max(y_true), np.max(preds)]))
                                fig_scatter.add_shape(type="line", x0=minv, x1=maxv, y0=minv, y1=maxv, line=dict(dash="dash", color="white"))
                                st.plotly_chart(fig_scatter, use_container_width=True)

                                # Tolerance-based "accuracy" (single float)
                                rng = maxv - minv
                                tol = 0.01 * rng if rng > 0 else 0.01
                                tol_acc = np.mean(np.abs(y_true - preds) <= tol)
                                st.caption(f"Tolerance-based accuracy (±{tol:.4g}): {tol_acc:.4f} ({tol_acc*100:.2f}%)")

                    else:
                        st.info("Column 'fs' not found. Predictions saved to 'fs_predicted' but no metrics computed.")

                    # Download predictions
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Manual Input
        st.subheader("Manual Data Entry")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
            hhis = st.number_input("HHIS", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
            hhig = st.number_input("HHIG", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
            hhic = st.number_input("HHIC", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
        
        with col2:
            hhit = st.number_input("HHIT", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
            ccr = st.number_input("CCR", min_value=0.0, max_value=1.0, value=0.7, format="%.1f")
            mcr = st.number_input("MCR", min_value=0.0, max_value=1.0, value=0.5, format="%.1f")
            asset_size = st.number_input("Asset Size", min_value=0, value=2000)
        
        with col3:
            ownership = st.selectbox("Ownership", [0, 1])
            covid = st.selectbox("COVID Period", [0, 1])
            inflation = st.number_input("Inflation Rate", min_value=0.0, max_value=1.0, value=0.05, format="%.3f")
            bank_age = st.number_input("Bank Age", min_value=0, value=100)
        
        # Optional: Add actual FS for accuracy calculation
        st.subheader("Optional: Actual Financial Stability Score")
        include_actual = st.checkbox("Include actual FS value for accuracy calculation")
        actual_fs = None
        if include_actual:
            actual_fs = st.number_input("Actual FS", min_value=0.0, value=2.0, format="%.1f")
        
        if st.button("Predict Financial Stability", type="primary"):
            # Prepare input data
            input_data = np.array([[year, hhis, hhig, hhic, hhit, ccr, mcr, 
                                  asset_size, ownership, covid, inflation, bank_age]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown(f'<div class="metric-container"><h2>Predicted Financial Stability Score: {prediction:.2f}</h2></div>', 
                       unsafe_allow_html=True)
            
            if include_actual and actual_fs is not None:
                accuracy = 1 if prediction == actual_fs else 0
                st.markdown(f'<div class="metric-container"><h3>Prediction Match: {"✅ Correct" if accuracy else "❌ Incorrect"}</h3></div>', 
                           unsafe_allow_html=True)

def linear_regression_page():
    st.markdown('<h2 class="sub-header">Linear Regression with Factor Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This section performs linear regression analysis using Factor Analysis for dimensionality reduction.
    Factor Analysis helps identify underlying relationships between variables while preserving individual
    correlations with the target variable.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file for regression analysis", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'fs' not in df.columns:
                st.error("Dataset must contain 'fs' column for regression analysis.")
                return
            
            # Check for required columns
            missing_cols = [col for col in COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return
            
            X = df[COLUMNS]
            y = df['fs']
            
            # Display correlation matrix
            st.subheader("Correlation Analysis")
            corr_with_target = X.corrwith(y).sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_corr = px.bar(x=corr_with_target.values, 
                                y=corr_with_target.index,
                                orientation='h',
                                title="Correlation with Financial Stability",
                                labels={'x': 'Correlation Coefficient', 'y': 'Variables'})
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("Correlation Coefficients")
                corr_df = pd.DataFrame({
                    'Variable': corr_with_target.index,
                    'Correlation': corr_with_target.values
                })
                st.dataframe(corr_df, use_container_width=True)
            
            # Factor Analysis
            st.subheader("Factor Analysis Configuration")
            n_factors = st.slider("Number of Factors", min_value=1, max_value=len(COLUMNS), value=min(5, len(COLUMNS)))
            
            if st.button("Perform Factor Analysis & Linear Regression", type="primary"):
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Factor Analysis
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                X_factors = fa.fit_transform(X_scaled)
                
                # Linear Regression on factors
                lr = LinearRegression()
                lr.fit(X_factors, y)
                
                # Predictions
                y_pred = lr.predict(X_factors)
                
                # Calculate metrics
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MAE", f"{mae:.4f}")
                with col4:
                    st.metric("MSE", f"{mse:.4f}")
                
                # Factor loadings
                st.subheader("Factor Loadings")
                loadings = pd.DataFrame(
                    fa.components_.T,
                    columns=[f'Factor_{i+1}' for i in range(n_factors)],
                    index=COLUMNS
                )
                st.dataframe(loadings, use_container_width=True)
                
                # Regression equation
                st.subheader("Linear Regression Equation")
                equation_parts = [f"{lr.intercept_:.4f}"]
                for i, coef in enumerate(lr.coef_):
                    sign = "+" if coef >= 0 else ""
                    equation_parts.append(f"{sign}{coef:.4f} × Factor_{i+1}")
                
                equation = "FS = " + " ".join(equation_parts)
                
                st.markdown(f'<div class="equation-box"><h4>Regression Equation:</h4><p style="font-family: monospace; font-size: 1.1em;">{equation}</p></div>', 
                           unsafe_allow_html=True)
                
                # Factor coefficients
                st.subheader("Factor Coefficients")
                coef_df = pd.DataFrame({
                    'Factor': [f'Factor_{i+1}' for i in range(n_factors)],
                    'Coefficient': lr.coef_,
                    'Abs_Coefficient': np.abs(lr.coef_)
                }).sort_values('Abs_Coefficient', ascending=False)
                
                fig_coef = px.bar(coef_df, x='Factor', y='Coefficient',
                                title="Factor Coefficients in Linear Regression")
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Prediction vs Actual plot
                st.subheader("Prediction vs Actual Values")
                fig_pred = px.scatter(x=y, y=y_pred, 
                                    labels={'x': 'Actual FS', 'y': 'Predicted FS'},
                                    title="Predicted vs Actual Financial Stability")
                fig_pred.add_shape(type="line", x0=y.min(), y0=y.min(), 
                                 x1=y.max(), y1=y.max(),
                                 line=dict(dash="dash", color="red"))
                st.plotly_chart(fig_pred, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in regression analysis: {str(e)}")

def performance_page(model):
    st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a test dataset to evaluate the model's performance across various metrics.
    The dataset should contain all required features and the target variable 'fs'.
    """)
    
    uploaded_file = st.file_uploader("Upload test dataset (CSV)", type="csv")
    
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
        mean_squared_error, mean_absolute_error, r2_score
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if 'fs' not in df.columns:
                st.error("Dataset must contain 'fs' column for performance evaluation.")
                return

            # Check for required feature columns
            missing_cols = [col for col in COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return

            X = df[COLUMNS]
            y_true = np.asarray(df['fs']).ravel()

            st.success(f"Dataset loaded: {len(df)} samples")

            if st.button("Evaluate Model Performance", type="primary"):
                try:
                    y_pred_raw = model.predict(X)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    raise

                y_pred = np.asarray(y_pred_raw).ravel()

                # Align sizes if necessary
                if len(y_true) != len(y_pred):
                    minlen = min(len(y_true), len(y_pred))
                    y_true = y_true[:minlen]
                    y_pred = y_pred[:minlen]
                    st.warning(f"Trimmed to first {minlen} rows to match true/predicted lengths.")

                n_samples = len(y_true)
                if n_samples < 3:
                    st.info("Metrics not calculated for datasets with less than 3 rows.")
                    # still show basic table
                    df_result = df.copy()
                    df_result['fs_predicted'] = pd.Series(y_pred)
                    st.subheader("Predictions (trimmed)")
                    st.dataframe(df_result, use_container_width=True)
                    return

                # Heuristic to decide classification vs regression
                unique_true = np.unique(y_true)
                unique_pred = np.unique(y_pred)
                n_unique = max(len(unique_true), len(unique_pred))
                is_integer_like = np.all(np.mod(y_true, 1) == 0) and np.all(np.mod(y_pred, 1) == 0)
                has_model_classes = hasattr(model, "classes_")
                is_classification = has_model_classes or is_integer_like or (n_unique <= 10)

                st.subheader("Overall Performance")
                col1, col2, col3 = st.columns(3)

                if is_classification:
                    # encode labels to integer codes with consistent ordering
                    labels = np.unique(np.concatenate([unique_true, unique_pred]))
                    y_true_enc = pd.Categorical(y_true, categories=labels).codes
                    y_pred_enc = pd.Categorical(y_pred, categories=labels).codes

                    acc = accuracy_score(y_true_enc, y_pred_enc)
                    correct_predictions = int(np.sum(y_true_enc == y_pred_enc))

                    with col1:
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc*100:.2f}%")
                    with col2:
                        st.metric("Total Samples", n_samples)
                    with col3:
                        st.metric("Correct Predictions", correct_predictions)

                    # Confusion matrix (show original label names on axes)
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true_enc, y_pred_enc)
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
                    fig_cm.update_xaxes(tickvals=list(range(len(labels))), ticktext=[str(l) for l in labels])
                    fig_cm.update_yaxes(tickvals=list(range(len(labels))), ticktext=[str(l) for l in labels])
                    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Classification report
                    st.subheader("Detailed Classification Report")
                    report = classification_report(y_true_enc, y_pred_enc, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    # map integer label indices to original labels if present
                    label_map = {str(i): str(labels[i]) for i in range(len(labels))}
                    new_index = []
                    for idx in report_df.index:
                        if idx.isdigit():
                            new_index.append(label_map.get(idx, idx))
                        else:
                            new_index.append(idx)
                    report_df.index = new_index
                    report_df = report_df.round(4)
                    st.dataframe(report_df, use_container_width=True)

                    # Distribution plots
                    st.subheader("Prediction Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_actual = px.histogram(y_true, title="Actual FS Distribution",
                                                  labels={'value': 'Financial Stability Score', 'count': 'Frequency'})
                        st.plotly_chart(fig_actual, use_container_width=True)
                    with col2:
                        fig_pred = px.histogram(y_pred, title="Predicted FS Distribution",
                                                labels={'value': 'Financial Stability Score', 'count': 'Frequency'})
                        st.plotly_chart(fig_pred, use_container_width=True)

                else:
                    # Regression metrics
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = float(np.sqrt(mse))
                    r2 = r2_score(y_true, y_pred)
                    correct_predictions = None  # meaningless for regression

                    with col1:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col2:
                        st.metric("Total Samples", n_samples)
                    with col3:
                        st.metric("MAE", f"{mae:.4f}")

                    # Display regression summary
                    st.markdown(
                        f"**MSE:** {mse:.4f} &nbsp;&nbsp; **R²:** {r2:.4f}"
                    )

                    # Prediction distribution and scatter
                    st.subheader("Prediction Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_actual = px.histogram(y_true, title="Actual FS Distribution",
                                                  labels={'value': 'Financial Stability Score', 'count': 'Frequency'})
                        st.plotly_chart(fig_actual, use_container_width=True)
                    with col2:
                        fig_pred = px.histogram(y_pred, title="Predicted FS Distribution",
                                                labels={'value': 'Financial Stability Score', 'count': 'Frequency'})
                        st.plotly_chart(fig_pred, use_container_width=True)

                    st.subheader("True vs Predicted")
                    fig_scatter = px.scatter(x=y_true, y=y_pred, labels={'x': 'True', 'y': 'Predicted'}, title='True vs Predicted')
                    minv = float(np.min([y_true.min(), y_pred.min()]))
                    maxv = float(np.max([y_true.max(), y_pred.max()]))
                    fig_scatter.add_shape(type="line", x0=minv, x1=maxv, y0=minv, y1=maxv, line=dict(dash="dash", color = "white"))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # Error analysis
                    st.subheader("Error Analysis")
                    errors = np.abs(y_true - y_pred)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error", f"{np.mean(errors):.4f}")
                        st.metric("Max Error", f"{np.max(errors):.4f}")
                    with col2:
                        st.metric("Std of Errors", f"{np.std(errors):.4f}")
                        st.metric("Min Error", f"{np.min(errors):.4f}")
                    fig_error = px.histogram(errors, title="Distribution of Absolute Errors",
                                            labels={'value': 'Absolute Error', 'count': 'Frequency'})
                    st.plotly_chart(fig_error, use_container_width=True)

                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': COLUMNS,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                    st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error in performance evaluation: {str(e)}")
    
    # Model Information
    st.subheader("Model Information")
    st.markdown("""
    **Model Type**: XGBoost Classifier  
    **Model Path**: `models/xgb_best.joblib`  
    **Features Used**: 12 financial and temporal indicators  
    **Target Variable**: Financial Stability Score (fs)  
    """)
    
    # Column Information
    with st.expander("📋 Feature Descriptions"):
        desc_df = pd.DataFrame(list(COLUMN_DESCRIPTIONS.items()), 
                              columns=['Feature', 'Description'])
        st.dataframe(desc_df, use_container_width=True)

if __name__ == "__main__":
    main()
