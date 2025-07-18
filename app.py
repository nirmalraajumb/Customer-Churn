
import gradio as gr
import joblib
import numpy as np

# Load the saved model
model = joblib.load("random_forest_model.joblib")

# Prediction function
def predict_churn(age, gender, tenure, usage, support_calls, payment_delay, subscription_type, contract_length, total_spend, last_interaction):
    # Encode inputs
    gender_encoded = 0 if gender == "Female" else 1
    subscription_map = {"Basic": 1, "Standard": 2, "Premium": 3}
    contract_map = {"Monthly": 1, "Quarterly": 2, "Annual": 3}
    
    input_data = np.array([[age, gender_encoded, tenure, usage, support_calls, payment_delay,
                            subscription_map[subscription_type], contract_map[contract_length],
                            total_spend, last_interaction]])
    
    prediction = model.predict(input_data)[0]
    return "Churn" if prediction == 1 else "Not Churn"

# Gradio interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Female", "Male"], label="Gender"),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Usage Frequency"),
        gr.Number(label="Support Calls"),
        gr.Number(label="Payment Delay"),
        gr.Radio(["Basic", "Standard", "Premium"], label="Subscription Type"),
        gr.Radio(["Monthly", "Quarterly", "Annual"], label="Contract Length"),
        gr.Number(label="Total Spend"),
        gr.Number(label="Last Interaction (days ago)")
    ],
    outputs="text",
    title="Customer Churn Predictor",
    description="Predict whether a customer is likely to churn based on their activity and account details."
)

iface.launch()
