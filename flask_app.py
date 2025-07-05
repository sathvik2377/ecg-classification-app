from flask import Flask, render_template, request, redirect, url_for, send_file
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import io
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Label meanings dictionary
label_meaning = {
    'N': 'Normal / Non-ectopic beat',
    'S': 'Supraventricular ectopic beat',
    'V': 'Ventricular ectopic beat',
    'F': 'Fusion beat',
    'Q': 'Unknown beat'
}

# Global variables to store model and data
model = None
predictions = None
df_processed = None

def gaussian_wave(center, width, amplitude, t):
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

def plot_synthetic_ecg(features, row_index, prediction):
    fs = 500
    rr_interval = features['RR_interval']
    num_samples = int(rr_interval * fs)
    t = np.linspace(0, rr_interval, num_samples)

    p_center = 0.2 * rr_interval
    q_center = 0.4 * rr_interval
    r_center = 0.42 * rr_interval
    s_center = 0.44 * rr_interval
    t_center = 0.6 * rr_interval

    p_wave = gaussian_wave(p_center, 0.025, features['P_amp'], t)
    q_wave = gaussian_wave(q_center, 0.012, features['Q_amp'], t)
    r_wave = gaussian_wave(r_center, 0.015, features['R_amp'], t)
    s_wave = gaussian_wave(s_center, 0.012, features['S_amp'], t)
    t_wave = gaussian_wave(t_center, 0.05, features['T_amp'], t)

    ecg_waveform = p_wave + q_wave + r_wave + s_wave + t_wave

    plt.figure(figsize=(10, 4))
    plt.plot(t, ecg_waveform, label=f'Beat {row_index}', color='blue', linewidth=2)
    plt.axvline(p_center, color='green', linestyle='--', alpha=0.7, label='P')
    plt.axvline(q_center, color='purple', linestyle='--', alpha=0.7, label='Q')
    plt.axvline(r_center, color='red', linestyle='--', alpha=0.7, label='R')
    plt.axvline(s_center, color='orange', linestyle='--', alpha=0.7, label='S')
    plt.axvline(t_center, color='brown', linestyle='--', alpha=0.7, label='T')
    
    prediction_text = f"{prediction} ({label_meaning.get(prediction, 'Unknown')})"
    plt.title(f"ECG Waveform - Beat {row_index} | Prediction: {prediction_text}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model is not None, 
                         predictions_made=predictions is not None)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model
    
    if 'model_file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['model_file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.pkl'):
        try:
            model = pickle.load(file)
            return render_template('index.html', model_loaded=True, 
                                 model_success="Model loaded successfully!",
                                 predictions_made=predictions is not None)
        except Exception as e:
            return render_template('index.html', model_loaded=False, 
                                 model_error=f"Error loading model: {str(e)}",
                                 predictions_made=predictions is not None)
    
    return redirect(url_for('index'))

@app.route('/upload_data', methods=['POST'])
def upload_data():
    global predictions, df_processed
    
    if model is None:
        return render_template('index.html', model_loaded=False, 
                             data_error="Please upload a model first!",
                             predictions_made=False)
    
    if 'data_file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['data_file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return render_template('index.html', model_loaded=True, 
                                 data_error="Please upload a CSV or Excel file",
                                 predictions_made=False)
        
        # Process data
        df_processed = df.drop(columns=['Annotation', 'Annotation_meaning', 'aami_label'], errors='ignore')
        
        # Get required features
        required_features = model.feature_names_in_
        
        # Add missing columns
        missing_cols = []
        for col in required_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
                missing_cols.append(col)
        
        # Reorder columns to match training
        df_processed = df_processed[required_features]
        
        # Make predictions
        predictions = model.predict(df_processed)
        
        # Calculate summary
        summary = Counter(predictions)
        
        return render_template('results.html', 
                             predictions=predictions,
                             summary=summary,
                             label_meaning=label_meaning,
                             data_length=len(df_processed),
                             missing_cols=missing_cols)
        
    except Exception as e:
        return render_template('index.html', model_loaded=True, 
                             data_error=f"Error processing data: {str(e)}",
                             predictions_made=False)

@app.route('/plot/<int:beat_index>')
def plot_ecg(beat_index):
    if df_processed is None or predictions is None:
        return "No data available", 404
    
    if beat_index >= len(df_processed):
        return "Beat index out of range", 404
    
    plot_url = plot_synthetic_ecg(df_processed.iloc[beat_index], beat_index, predictions[beat_index])
    return render_template('plot.html', plot_url=plot_url, beat_index=beat_index,
                         prediction=predictions[beat_index],
                         meaning=label_meaning.get(predictions[beat_index], 'Unknown'))

@app.route('/download_results')
def download_results():
    if predictions is None:
        return "No results available", 404

    results_df = pd.DataFrame({
        'Beat_Index': range(len(predictions)),
        'Prediction': predictions,
        'Meaning': [label_meaning.get(p, 'Unknown') for p in predictions]
    })

    output = io.StringIO()
    results_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ecg_predictions.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, port=8507)
