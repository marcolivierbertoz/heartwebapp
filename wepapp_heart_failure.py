# Loading packages ##########################################################################
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import requests,json

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder





###############################################################################################
st.set_page_config(layout="wide")

# Creating std scaler
# sc = StandardScaler()

# # Loading the model saved
# model = keras.models.load_model("Test_model_Neural_Network")

# Loading of the data
@st.cache
def load_data():
    data=pd.read_csv('heart_failure_clinical_records_dataset.csv')
    return data

df = load_data()

# Creating functions for working with data & list #####################################################
# List for prediction
# pred_label = list()

# # Defining data preparation
# def preparazione(dati_originali):
#     dati_pronti = dati_originali.values
#     dati_pronti = sc.fit_transform(dati_originali)
#     return dati_pronti

# # Defining function for prediction
# def predict(dati):
#     previsione = model.predict(dati)
#     # Adding to list
#     for i in range(len(previsione)):
#         pred_label.append(np.argmax(previsione[i]))
#     return pred_label



##########################################################################################

# Sidebar creation ####################################################################
st.sidebar.title('Patient Parameters')

# Creating parameters
params = {
    'Age':st.sidebar.slider('Age', min(df['age']), max(df['age']),1.0),
    'Anemia':st.sidebar.selectbox('Anemia', ('Yes','No')),
    'CPK':st.sidebar.slider('CPK [mcg/L]', min(df['creatinine_phosphokinase']), max(df['creatinine_phosphokinase']), 1),
    'Diabetes':st.sidebar.selectbox('Diabetes', ('Yes','No')),
    'Ejection fraction':st.sidebar.slider('Ejection fraction [%]', min(df['ejection_fraction']), max(df['ejection_fraction']), 1),
    'High Blood Pressure':st.sidebar.selectbox('High Blood Pressure', ('Yes','No')),
    'Platelets in the blood':st.sidebar.slider('Platelets in the blood [kiloplatelets/mL]', min(df['platelets']), max(df['platelets']),1.0),
    'Serum Creatinine':st.sidebar.slider('Serum Creatinine [mg/dL]', min(df['serum_creatinine']), max(df['serum_creatinine']),1.0),
    'Serum Sodium':st.sidebar.slider('Serum Sodium [mEq/L]', min(df['serum_sodium']), max(df['serum_sodium']),1),
    'Woman or Man':st.sidebar.selectbox('Woman or Man', ('Woman','Men')),
    'Smoking':st.sidebar.selectbox('Smoking',('Yes','No')),
    'Time':st.sidebar.slider('Time [days]', min(df['time']), max(df['time']),1)

}

# Method for changing attribute
def change_type(X):
    if X == 'Yes':
        X = 1
    else:
        X = 0    
    return X

def change_sex(X):
    if X == 'Woman':
        X = 1
    else:
        X = 0
    return X            

# Defining user input
def user_input():
    age = params['Age']
    anemia = params['Anemia']
    cpk = params['CPK']
    diabetes = params['Diabetes']
    ejection_fraction = params['Ejection fraction']
    hbp = params['High Blood Pressure']
    platelets = params['Platelets in the blood']
    serum_creatinine = params['Serum Creatinine']
    serum_sodium = params['Serum Sodium']
    woman_man = params['Woman or Man']
    smoking = params['Smoking']
    time = params['Time']

    data = {
        'Age': age,
        'Anemia': change_type(anemia),
        'CPK': cpk,
        'Diabetes':change_type(diabetes),
        'Ejection_fraction':ejection_fraction,
        'HBP':change_type(hbp),
        'Platelets':platelets,
        'Serum_Creatinine':serum_creatinine,
        'Serum_Sodium':serum_sodium,
        'Woman_Man':change_sex(woman_man),
        'Smoking':change_type(smoking),
        'Time':time
    }

    features = pd.DataFrame(data, index=[0])

    return features

# API information
url = 'https://heart-api-first.herokuapp.com'
endpoint = '/prediction'


# defining process for calling API
def process(data, server_url: str):
    heart_data = data

    result = requests.post(server_url,data = heart_data)

    return result


# Expander ##################################################################################
with st.beta_expander('Information', expanded=True):
    st.header('Goal of the Web Application')
    st.write('The main goal of this project to learn on how to use the fastAPI package for creating background services, in this case, a machine learning model.')
    st.write('**Source of data**: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data')
    st.write("""
    > **Attention**: 
    * This app is **NOT** meant as a medical device/applicaiton for medical diagnosis; 
    * It is just for educational purpose;
    * If you think you have an health problem, please reach out to an authorized medical professional;
    * The creator of this applicaiton will **NOT** assume any responsability for any misuse of the application.
    """)
    st.header('Parameters Information')
    st.write("""
    * **Age**: Age of the patient;
    * **Anemia**: Decrease of red blood cells or hemoglobin;
    * **Creatinine Phosphokinase Enzyme (CPK)**: Level of CPK enzyme in the blood (mcg/L);
    * **Diabetes**: Patient is diabetic or not;
    * **Ejection Fraction**: Percentage of blood that leaves the heart at each contraction;
    * **High Blood Pressure**: Patient has high blood pressure;
    * **Platelets in the blood**: Number of platelets in the blood [kiloplatelets/mL];
    * **Serum Creatinine**: Level of serum creatinine in the blood [mg/dL];
    * **Serum Sodium**: Level of serum sodium in the blood [mEq/L];
    * **Woman or Man**: If patient is a woman or man;
    * **Smoking**: If patient smokes or not;
    * **Time**: Follow-up periods, number of days after the first check.
    """)

    

# Columns creation #####################################################################
left_column1, right_column1 = st.beta_columns(2)

with left_column1:
    st.header('Heart Animation')
    st.write('In this GIF, you can see how the heart works, with the name of its major components.')
    components.iframe("https://healthblog.uofmhealth.org/sites/consumer/files/2020-01/heart_pumping.gif",height=500)
with right_column1:
    st.header('Heart Attack Video')
    st.write('In this video, created by the YouTube channel _Institute of Human Anatomy_, explain what happens during a heart attack.')
    st.write('**Attention**: Video contains content (real human heart from a dead body) that may cause some problem for some people.')
    st.video("https://www.youtube.com/watch?v=m0fiHGlqivo&t=1s")


# Creating the data  ####################################################################

user_input_original_df = user_input()
user_input_original_df




# Creating button
# if st.sidebar.button("Predict"):
#     # Preparing data
#     dati_preparati = preparazione(user_input_original_df)

#     #Predicting data
#     prediction_heart = predict(dati_preparati)
   
    
#     # Viewing the reult
#     prediction_heart
# else:
#     pass    


# test_df = pd.DataFrame(user_input_original_df).dtypes
# test_df




if st.sidebar.button("Predict"):
    result = process(user_input_original_df, url+endpoint)
    result.text
else:
	pass







