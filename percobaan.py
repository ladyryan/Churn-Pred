# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:39 2022

@author: asus
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loadModel = pickle.load(open('modelpred.pkl', 'rb'))


@st.cache()

# creating a function for prediction
def churnpredict(input_data):
    
    # changing the input_data as array
    input_data_asarray = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_asarray.reshape(1,-1)

    prediction = loadModel.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not churned'
    else:
        return 'The person is churned'
    
    
def main():
    
    #giving a title
    st.title('Bank Customer Churn Prediction Web App')
    
    
    #getting the input data from the user
    Gender = st.text_input('Gender')
    Age = st.text_input('Age')
    CreditScore = st.text_input('Credit Score')
    EstimatedSalary = st.text_input('Salary')
    HasCrCard = st.text_input('Has Credit Card?')
    CrdScoreGivenAge = st.text_input('Credit Score Given Age')
    
    
    # code for Prediction
    analysis = ''
    
    # generating a button for prediction
    if st.button('Churn Prediction Result'):
        analysis = churnpredict([Gender, Age, CreditScore, EstimatedSalary,HasCrCard, CrdScoreGivenAge])
        
    st.success(analysis)
    
    
    

if __name__ == '__main__':
    main()