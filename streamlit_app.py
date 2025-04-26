import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."

    import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime

# Step 1: Load and Prepare Data
response = requests.get("https://api.thaistock2d.com/live")
data = response.text
lines = data.split('\n')

# Parse data for SET and Value
set_value = float(lines[1].split()[1].replace(",", ""))
value_million = float(lines[2].split()[1].replace(",", ""))

# Step 2: Example Historical Data for Model Training (Replace with your dataset)
data = {
    'date': ['2025-04-20', '2025-04-21', '2025-04-22'],
    'set': [1159.00, 1160.00, 1161.00],
    'value': [33120.92, 33125.12, 33130.50],
    '2d': ['00', '01', '02']
}

df = pd.DataFrame(data)

# Feature Engineering (previous 3 values of 2D)
df['prev_1'] = df['2d'].shift(1)
df['prev_2'] = df['2d'].shift(2)
df['prev_3'] = df['2d'].shift(3)
df = df.dropna()

# Prepare feature set and target variable
X = df[['prev_1', 'prev_2', 'prev_3']]  # Features
y = df['2d']  # Target variable (2D)

# Step 3: Train Model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict Next 2D Value (based on time prediction)
latest_data = X.tail(1)
predicted_2d = model.predict(latest_data)[0]

# Step 5: Time-based Forecasting (12:01 PM)
current_time = datetime.datetime.now()

# Check if current time is 12:01 PM
if current_time.hour == 12 and current_time.minute == 1:
    time_forecast = f"Forecasting at 12:01 PM, next 2D: {predicted_2d}"
else:
    time_forecast = f"Current time: {current_time.strftime('%H:%M:%S')} - No forecast at this time."

# Step 6: Streamlit UI
st.title("📊 Thai 2D Live Dashboard")

# Display current data
st.write(f"**SET Index:** {set_value}")
st.write(f"**Value (M):** {value_million}")
st.write(f"**Predicted 2D:** {predicted_2d}")
st.write(f"**Time-based Forecast:** {time_forecast}")

)
