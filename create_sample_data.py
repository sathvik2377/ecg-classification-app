import pandas as pd
import numpy as np

# Create sample ECG features
features = ["PR_interval", "QRS_duration", "QT_interval", "RR_interval", "RR_variation",
            "P_amp", "Q_amp", "R_amp", "S_amp", "T_amp", "PR_slope", "QR_slope", 
            "RS_slope", "ST_slope", "Pre_R_avg_amp", "Post_R_avg_amp", "Heart_rate", 
            "RR_SDNN", "RR_RMSSD"]

# Generate 10 random records
data = []
for i in range(10):
    record = {
        "PR_interval": np.random.uniform(0.12, 0.20),
        "QRS_duration": np.random.uniform(0.06, 0.10),
        "QT_interval": np.random.uniform(0.35, 0.45),
        "RR_interval": np.random.uniform(0.6, 1.2),
        "RR_variation": np.random.uniform(0, 0.1),
        "P_amp": np.random.uniform(-0.3, 0.3),
        "Q_amp": np.random.uniform(-0.5, 0),
        "R_amp": np.random.uniform(0.5, 2.0),
        "S_amp": np.random.uniform(-1.0, 0),
        "T_amp": np.random.uniform(-0.2, 0.5),
        "PR_slope": np.random.uniform(0.01, 0.05),
        "QR_slope": np.random.uniform(0.05, 0.15),
        "RS_slope": np.random.uniform(-0.2, -0.05),
        "ST_slope": np.random.uniform(0, 0.02),
        "Pre_R_avg_amp": np.random.uniform(-0.2, 0.2),
        "Post_R_avg_amp": np.random.uniform(-0.2, 0.2),
        "Heart_rate": np.random.uniform(60, 100),
        "RR_SDNN": np.random.uniform(0.05, 0.5),
        "RR_RMSSD": np.random.uniform(0.1, 0.8)
    }
    data.append(record)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('ecg_features.csv', index=False)

print("Sample ECG data saved as 'ecg_features.csv'")