import streamlit as st
import pickle

# Load model and vectorizer
model_path = "/content/drive/MyDrive/Colab Notebooks/p1 intern/sms-spam-detection/model.pkl"
vectorizer_path = "/content/drive/MyDrive/Colab Notebooks/p1 intern/sms-spam-detection/vectorizer.pkl"


with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("SMS Spam Detector")
st.write("Enter an SMS message below to check if it's spam or not.")

user_input = st.text_area("Enter your SMS message here:")

if st.button("Predict"):
    if user_input:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        result = "Spam" if prediction[0] == 1 else "Ham (Not Spam)"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message before predicting.")
