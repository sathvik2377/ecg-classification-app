# ECG Beat Classification System

A Streamlit web application for classifying ECG heartbeat types using machine learning.

## Features

- Upload trained machine learning models (.pkl format)
- Upload ECG feature data (.csv or .xlsx format)
- Visualize ECG waveforms
- Classify heartbeats into different categories (Normal, Supraventricular, Ventricular, etc.)
- Download classification results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ecg-classification-app.git
cd ecg-classification-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run index.py
```

## Usage

1. Upload a trained model file (.pkl)
2. Upload ECG feature data (.csv or .xlsx)
3. Click "Make Predictions" to classify the heartbeats
4. View the results and visualizations
5. Download the classification results as CSV

## Sample Data

The repository includes scripts to generate sample data:

- `create_sample_model.py`: Creates a sample classification model
- `create_sample_data.py`: Generates sample ECG feature data

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn

## License

[MIT](LICENSE)