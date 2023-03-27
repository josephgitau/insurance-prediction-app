# Model deployment using streamlit
import streamlit as st
import pandas as pd
import joblib

# load trained model
model = joblib.load('insurance_model_RF.joblib')

# define function to predict insurance cost
def predict_charges(input_data):
    return model.predict(input_data)

# Title 
st.title('Insurance Cost Prediction App')

# collect user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 30)
    sex = st.sidebar.selectbox("Sex", ("male", "female"))
    bmi = st.sidebar.slider('BMI', 15, 50, 25)
    children = st.sidebar.slider('Number of Children', 0, 10, 2)
    smoker = st.sidebar.selectbox("Smoker", ("yes", "no"))
    region = st.sidebar.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    return pd.DataFrame(
        {
            'age': [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        }
    )

input_df = user_input_features()

# display user input in a table
st.subheader('User Input Parameters')
st.write(input_df)

# make predictions and display the result
st.header('Prediction of Insurance Cost')
prediction = predict_charges(input_df)
st.write(f"Insurance Cost: ${prediction[0]:.2f}")
