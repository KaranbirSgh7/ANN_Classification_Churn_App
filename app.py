from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.preprocessing import OneHotEncoder


model=load_model('model.keras')

with open('one_hot_encoder_geo1.pickle','rb') as file:
    label_encode=pickle.load(file)

with open('scaler.pickle','rb') as file1:
    scalar=pickle.load(file1)

with open('label_encoder_gender.pickle','rb') as file:
    onHot=pickle.load(file)

categories_geo=label_encode.categories_[0]

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', categories_geo)
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])



input_data = {
    'CreditScore': credit_score,
    'Gender': onHot.transform([gender]),
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])

geography_new=label_encode.transform([[geography]]).toarray()

test1 = pd.DataFrame(
    geography_new,
    columns=label_encode.get_feature_names_out(['Geography'])
)


input_df=pd.concat([input_df,test1],axis=1)

final_Data=scalar.transform(input_df)

prediction=model.predict(final_Data)

st.write(f"Prediction is {prediction}")

st.progress(float(prediction[0][0]))

if(prediction[0][0]>0.5):
    st.write("Customer is likely to chern..")
else:
    st.write("Customer is not likely to chern..")



