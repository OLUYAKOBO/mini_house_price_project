import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model, label encoders, and scaler
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scal.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))

# Streamlit app
st.title('House Price Prediction')

st.header("*Enter the informations of the House below*")
def house_input():
# Collect user inputs
    house_age = st.number_input('House Age', min_value=1800, max_value=2020, value=2000)
    bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=8, value=3)
    bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=3, value=2)
    area = st.number_input('Area (in sq ft)', min_value=100, max_value=300000, value=150000)
    location = st.selectbox('Location', (['Urban', 'Suburban', 'Rural']))

    feat = np.array([house_age,bedrooms,bathrooms,area,location]).reshape(1,-1)
    cols = ['HouseAge','Bedroom','FullBath','LotArea','Location']

    df = pd.DataFrame(feat, columns = cols)
    return df
df = house_input()
#st.write(df)

def prepare(df):
    df1 = df.copy()
    cat_cols = ['Location']
    encoded_data = encoder.transform(df1[cat_cols])
    dense_data = encoded_data.todense()
    df1_encoded = pd.DataFrame(dense_data, columns = encoder.get_feature_names_out())

    df1 = pd.concat([df1,df1_encoded],
                    axis = 1)
    df1.drop(cat_cols,
             axis = 1,
             inplace = True)
    
    
    cols = df1.columns
    df1 = scaler.transform(df1)
    df1 = pd.DataFrame(df1,columns=cols)
    return df1
df1 = prepare(df)
#st.write(df1)

# Make prediction
predictions = model.predict(df1)
import time
st.subheader('*House Price*')
if st.button('*Click here to get the price of the **House***'):
    time.sleep(10)
    #st.write(predictions)
    st.write(f'The house is valued at {predictions.item()}')

# To run the Streamlit app, use the command:
# streamlit run app.py
