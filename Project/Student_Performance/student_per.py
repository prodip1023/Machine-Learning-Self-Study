import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder



def load_model():
    with open('model.pkl','rb') as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le



def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed
    
def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    hour_std = st.number_input('Hours Studied',max_value=10,min_value=1,value=5)
    prev_scores = st.number_input('Previous Scores',max_value=100,min_value=40,value=70)
    extra_act = st.selectbox("Extracurricular Activities",['Yes','No'])
    sleep_hrs = st.number_input('Sleep Hours',max_value=10,min_value=4,value=7)
    sample_paper_solved = st.number_input('Sample Question Papers Practiced',max_value=10,min_value=0,value=2)

    if st.button("Predict your score"):
        user_data = {'Hours Studied': hour_std, 
                     'Previous Scores': prev_scores, 
                     'Extracurricular Activities': extra_act, 
                     'Sleep Hours': sleep_hrs, 
                     'Sample Question Papers Practiced': sample_paper_solved
                     }
        prediction = predict_data(user_data)
        st.success(f"Your predicted performance index is: {prediction[0]:.2f}")




if __name__ == "__main__":
    main()