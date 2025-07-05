import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Create a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Sample features
features = ["PR_interval", "QRS_duration", "QT_interval", "RR_interval", 
            "P_amp", "Q_amp", "R_amp", "S_amp", "T_amp"]

# Create sample data
X = np.random.rand(100, len(features))
y = np.random.choice(['N', 'S', 'V', 'F', 'Q'], size=100)

# Train model
model.fit(X, y)

# Set feature names
model.feature_names_in_ = np.array(features)

# Save model
with open('ecg_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Sample model saved as 'ecg_model.pkl'")