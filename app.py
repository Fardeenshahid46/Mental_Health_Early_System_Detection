import gradio as gr
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ✅ Load model and encoders
try:
    model = joblib.load("risk_predictor_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    print("❌ Model or encoders not loaded:", e)
    raise

# ✅ Prediction function with file and table output
def predict_risk(sleep, study, stress, screen_time, activity, appetite, social, rested, relaxed):
    try:
        rested_encoded = label_encoders['rested'].transform([rested])[0]
        relaxed_encoded = label_encoders['relaxed'].transform([relaxed])[0]

        input_data = np.array([[sleep, study, stress, screen_time, activity,
                                appetite, social, rested_encoded, relaxed_encoded]])

        prediction = model.predict(input_data)[0]

        risk_mapping = {
            0: "🟢 Low Risk",
            1: "🟡 Moderate Risk",
            2: "🔴 High Risk"
        }
        readable_result = risk_mapping.get(prediction, "Unknown Risk Level")

        # ✅ Log new entry
        entry = {
            "datetime": datetime.now(),
            "sleep": sleep,
            "study": study,
            "stress": stress,
            "screen_time": screen_time,
            "activity": activity,
            "appetite": appetite,
            "social": social,
            "rested": rested,
            "relaxed": relaxed,
            "predicted_risk": readable_result
        }

        new_entry_df = pd.DataFrame([entry])
        log_file = "user_predictions_log.csv"

        # Append or create log file
        if os.path.exists(log_file):
            new_entry_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            new_entry_df.to_csv(log_file, index=False)

        # ✅ Read the full log to display
        full_log_df = pd.read_csv(log_file)

        return (
            readable_result,
            log_file,  # File path to be downloaded
            full_log_df.tail(10)  # Show last 10 records
        )

    except Exception as e:
        print("❌ Error during prediction:", e)
        return "Error", None, pd.DataFrame()

# ✅ Gradio UI
iface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(0, 12, step=1, label="Sleep Duration (hours/day)"),
        gr.Slider(0, 12, step=1, label="Study Hours"),
        gr.Slider(0, 10, step=1, label="Stress Level (1-10)"),
        gr.Slider(0, 12, step=1, label="Screen Time (hours/day)"),
        gr.Slider(0, 120, step=10, label="Physical Activity (minutes/day)"),
        gr.Slider(1, 5, step=1, label="Appetite Level (1-5)"),
        gr.Slider(0, 10, step=1, label="Social Interaction Level (1-10)"),
        gr.Radio(["Yes", "No"], label="Feeling Rested Today?"),
        gr.Radio(["Yes", "No"], label="Did You Relax Today?")
    ],
    outputs=[
        gr.Label(label="Predicted Mental Health Risk"),
        gr.File(label="Download Prediction Log (CSV)"),
        gr.Dataframe(label="Latest Predictions Log")
    ],
    title="🧠 Mental Health Early Alert System",
    description="Check your mental health risk level based on lifestyle inputs."
)

iface.launch()
