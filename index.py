import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ECG Beat Classification",
    page_icon="❤",
    layout="wide"
)

# Title and description
st.title("❤ ECG Beat Classification System")
st.write("Upload your trained model and ECG features to classify heartbeat types")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Label meanings dictionary
label_meaning = {
    'N': 'Normal / Non-ectopic beat',
    'S': 'Supraventricular ectopic beat',
    'V': 'Ventricular ectopic beat',
    'F': 'Fusion beat',
    'Q': 'Unknown beat'
}

# Gaussian wave function
def gaussian_wave(center, width, amplitude, t):
    try:
        return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
    except Exception as e:
        st.error(f"Error in gaussian_wave: {e}")
        return np.zeros_like(t)

# ECG plot function
def plot_synthetic_ecg(features, row_index, prediction):
    try:
        fs = 500
        rr_interval = float(features['RR_interval'])
        num_samples = int(rr_interval * fs)
        t = np.linspace(0, rr_interval, num_samples, dtype=np.float64)

        p_center = 0.2 * rr_interval
        q_center = 0.4 * rr_interval
        r_center = 0.42 * rr_interval
        s_center = 0.44 * rr_interval
        t_center = 0.6 * rr_interval

        p_wave = gaussian_wave(p_center, 0.025, float(features['P_amp']), t)
        q_wave = gaussian_wave(q_center, 0.012, float(features['Q_amp']), t)
        r_wave = gaussian_wave(r_center, 0.015, float(features['R_amp']), t)
        s_wave = gaussian_wave(s_center, 0.012, float(features['S_amp']), t)
        t_wave = gaussian_wave(t_center, 0.05, float(features['T_amp']), t)

        ecg_waveform = p_wave + q_wave + r_wave + s_wave + t_wave

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, ecg_waveform, label=f'Beat {row_index}', color='blue', linewidth=2)
        ax.axvline(p_center, color='green', linestyle='--', alpha=0.7, label='P')
        ax.axvline(q_center, color='purple', linestyle='--', alpha=0.7, label='Q')
        ax.axvline(r_center, color='red', linestyle='--', alpha=0.7, label='R')
        ax.axvline(s_center, color='orange', linestyle='--', alpha=0.7, label='S')
        ax.axvline(t_center, color='brown', linestyle='--', alpha=0.7, label='T')

        prediction_text = f"{prediction} ({label_meaning.get(prediction, 'Unknown')})"
        ax.set_title(f"ECG Waveform - Beat {row_index} | Prediction: {prediction_text}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        return fig
    except Exception as e:
        st.error(f"Error creating ECG plot: {e}")
        # Return a simple fallback plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error plotting Beat {row_index}\nPrediction: {prediction}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"ECG Waveform - Beat {row_index} (Error)")
        return fig

# Sidebar for file uploads
st.sidebar.header("📁 File Uploads")

# Step 1: Model Upload
st.sidebar.subheader("1. Upload Trained Model")
model_file = st.sidebar.file_uploader(
    "Choose your trained model file (.pkl)",
    type=['pkl'],
    help="Upload the pickle file containing your trained ECG classification model"
)

if model_file is not None:
    try:
        # Load the model with error handling
        model_file.seek(0)  # Reset file pointer
        model = pickle.load(model_file)
        st.session_state.model = model
        st.sidebar.success("✅ Model loaded successfully!")

        # Display model info
        if hasattr(model, 'feature_names_in_'):
            st.sidebar.info(f"Model expects {len(model.feature_names_in_)} features")
        else:
            st.sidebar.info("Model loaded (feature names not available)")

    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.sidebar.info("Try using a model saved with a compatible scikit-learn version")

# Step 2: ECG Data Upload
st.sidebar.subheader("2. Upload ECG Features")
data_file = st.sidebar.file_uploader(
    "Choose ECG feature file (.csv or .xlsx)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload the file containing ECG features for classification"
)

# Main content
if st.session_state.model is None:
    st.info("👆 Please upload your trained model file (.pkl) in the sidebar to get started")
elif data_file is None:
    st.info("👆 Please upload your ECG features file (.csv or .xlsx) in the sidebar")
else:
    try:
        # Load ECG data
        if data_file.name.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
        
        st.success(f"✅ Loaded ECG data: {len(df)} records")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head())
        
        # Process data and make predictions
        if st.button("🔮 Make Predictions", type="primary"):
            with st.spinner("Processing ECG data and making predictions..."):
                # Clean data
                df_processed = df.copy()

                # Drop annotation columns if they exist
                df_processed = df_processed.drop(columns=['Annotation', 'Annotation_meaning', 'aami_label'], errors='ignore')

                # Get required features from model
                if hasattr(st.session_state.model, 'feature_names_in_'):
                    required_features = st.session_state.model.feature_names_in_
                else:
                    # Fallback feature list
                    required_features = ['RR_interval', 'P_amp', 'Q_amp', 'R_amp', 'S_amp', 'T_amp']

                # Add missing columns with default values
                missing_cols = []
                for col in required_features:
                    if col not in df_processed.columns:
                        df_processed[col] = 0.0
                        missing_cols.append(col)

                if missing_cols:
                    st.warning(f"⚠ Added missing columns with default values: {', '.join(missing_cols)}")

                # Ensure all columns are numeric
                for col in required_features:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0.0)

                # Reorder columns to match training
                df_processed = df_processed[required_features]

                # Make predictions with error handling
                try:
                    predictions = st.session_state.model.predict(df_processed.values)
                except Exception as pred_error:
                    st.error(f"Prediction error: {pred_error}")
                    st.info("Trying alternative prediction method...")
                    predictions = st.session_state.model.predict(df_processed)
                
                # Store results in session state
                st.session_state.predictions = predictions
                st.session_state.df_processed = df_processed
                st.session_state.predictions_made = True
        
        # Display results if predictions are made
        if st.session_state.predictions_made:
            predictions = st.session_state.predictions
            df_processed = st.session_state.df_processed
            
            # Summary
            st.header("📈 Prediction Results")
            summary = Counter(predictions)
            
            # Create columns for summary display
            cols = st.columns(len(summary))
            for i, (label, count) in enumerate(summary.items()):
                with cols[i]:
                    meaning = label_meaning.get(label, 'Unknown')
                    st.metric(
                        label=f"{label} - {meaning}",
                        value=count,
                        help=f"Number of {meaning.lower()} detected"
                    )
            
            # Detailed results table
            with st.expander("📋 Detailed Results"):
                results_df = pd.DataFrame({
                    'Beat_Index': range(len(predictions)),
                    'Prediction': predictions,
                    'Meaning': [label_meaning.get(p, 'Unknown') for p in predictions]
                })
                st.dataframe(results_df)
            
            # ECG Waveform Visualization
            st.header("📊 ECG Waveform Visualization")
            
            # Allow user to select number of plots
            max_plots = min(10, len(df_processed))
            
            if max_plots == 1:
                num_plots = 1
                st.info("📊 Displaying the single ECG waveform in your dataset")
            else:
                num_plots = st.slider("Number of ECG waveforms to display:", 1, max_plots, min(5, max_plots))
            
            # Plot ECG waveforms
            for i in range(num_plots):
                prediction = predictions[i]
                st.subheader(f"Beat {i} - Prediction: {prediction} ({label_meaning.get(prediction, 'Unknown')})")
                
                try:
                    fig = plot_synthetic_ecg(df_processed.iloc[i], i, prediction)
                    st.pyplot(fig)
                    plt.close()  # Close figure to free memory
                except Exception as e:
                    st.error(f"Error plotting beat {i}: {str(e)}")
            
            # Download results
            st.header("💾 Download Results")
            results_df = pd.DataFrame({
                'Beat_Index': range(len(predictions)),
                'Prediction': predictions,
                'Meaning': [label_meaning.get(p, 'Unknown') for p in predictions]
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="ecg_predictions.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*ECG Beat Classification System* | Built with Streamlit ❤")