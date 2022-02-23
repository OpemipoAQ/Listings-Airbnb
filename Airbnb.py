
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.write("""
# AIRBNB HOUSE PRICE PREDICTION APP
""")
image=Image.open('Airbnb.jpg')
st.image(image, width=500)

model=pickle.load(open('model(rf).pkl','rb'))

scaler=pickle.load(open('scaler.pkl','rb'))


st.sidebar.header('User Input Parameters')

def user_input_features():
    Room_type=st.sidebar.selectbox('Room Type',('Private room','Entire home/apt','Shared room'))
    if Room_type=='Entire home/apt':
        Entire=1
        Private=0
        Shared=0
    if Room_type=='Private room':
        Entire=0
        Private=1
        Shared=0
    if Room_type=='Shared room':
        Entire=0
        Private=0
        Shared=1
        
    Region_hood=st.sidebar.selectbox('Region',('North Region','Central Region','East Region','West Region','North_East Region'))
    if Region_hood=='North Region':
        Central=0
        North=1
        East=0
        West=0
        North_East=0
    if Region_hood=='Central Region':
        Central=1
        North=0
        East=0
        West=0
        North_East=0
    if Region_hood=='East Region':
        Central=0
        North=0
        East=1
        West=0
        North_East=0
    if Region_hood=='West Region':
        Central=0
        North=0
        East=0
        West=1
        North_East=0
    if Region_hood=='North_East Region':
        Central=0
        North=0
        East=0
        West=0
        North_East=1
        
    Host_id=st.number_input('What is your Host ID?')
    host_list_count=st.number_input('Host listing count')
    longitude=st.number_input("Building's longitudinal location")
    latitude=st.number_input("Building's latitudinal location")    
    minimum_nights=st.number_input('For how many nights will you be staying?')
    Availability=st.number_input('how many nights is the building available')
    last_review_month=st.number_input('What month was the last review?',max_value=12,min_value=1,step=1)
    last_review_year=st.number_input('What year was the last review?',max_value=2022,min_value=2012,step=1)
    reviews_per_month=st.number_input('How many reviews do you get monthly?')
    number_of_reviews=st.number_input('How many reviews have you gotten?')
        
        
    data={'latitude':latitude,
         'longitude':longitude,
         'minimum_nights':minimum_nights,
         'calculated_host_listings_count':host_list_count,
         'last_review_month':last_review_month,
         'last_review_year':last_review_year,
         'reviews_per_month':reviews_per_month,
         'number_of_reviews':number_of_reviews,
         'host_id':Host_id,
         'availability_365':Availability,
         'room_type_Private_room':Private,
         'room_type_Shared_room':Shared,
         'neighbourhood_group_West Region':West,
         'neighbourhood_group_North-East Region':North_East,
         'neighbourhood_group_East Region':East,
         'neighbourhood_group_North Region':North}
    
    features=pd.DataFrame(data, index=[0])
    return features

input_df=user_input_features()
input_df=scaler.transform(input_df)
    
if st.button('PREDICT'):
    y_outcome=model.predict(input_df)
    st.write(f' This room will cost you $',y_outcome[0])
