import streamlit as st
import joblib
import os

# Load model and vectorizer
model_path = os.path.abspath("models/career_predictor.pkl")
vectorizer_path = os.path.abspath("models/vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Streamlit UI
st.set_page_config(page_title="Pixi - Career Guide", page_icon="🎓")

st.title("🎓 Pixi - AI Career Counselor")
st.write("Tell me about your interests and I’ll suggest a suitable career for you!")

user_input = st.text_area("💬 What do you love doing?", height=100)

if st.button("🔍 Recommend Career"):
    if user_input.strip() != "":
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        st.success(f"🎯 Based on your input, you might be a great **{prediction}**!")
    else:
        st.warning("Please enter something about your interests.")
