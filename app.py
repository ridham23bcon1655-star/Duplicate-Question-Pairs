
import streamlit as st
import pickle
import gzip

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("cv.pkl", "rb") as f:
    cv = pickle.load(f)

st.title("Duplicate Question Pair Detection")
st.write("Check whether two questions are duplicates.")

q1 = st.text_input("Enter First Question")
q2 = st.text_input("Enter Second Question")

if st.button("Check Duplicate"):
    if q1.strip() and q2.strip():
        features = cv.transform([q1 + " " + q2])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("Duplicate Questions")
        else:
            st.info("Not Duplicate Questions")
    else:
        st.warning("Please enter both questions.")

