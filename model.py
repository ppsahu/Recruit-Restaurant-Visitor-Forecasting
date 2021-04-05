# importing all the required libraries
import numpy as np
import pandas as pd
from datetime import datetime
import time, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from chart_studio.plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import math
from tqdm import tqdm

# Reading all the files
air_visit_data = pd.read_csv('air_visit_data.csv')
air_store_info = pd.read_csv('air_store_info.csv')
air_reserve = pd.read_csv('air_reserve.csv')
hpg_store_info = pd.read_csv('hpg_store_info.csv')
hpg_reserve = pd.read_csv('hpg_reserve.csv')
date_info = pd.read_csv('date_info.csv')
store_id_relation = pd.read_csv('store_id_relation.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# error metric
# kaggle

def root_mean_squared_logarithmic_error(p,a):  
    err=0
    for i in range(len(p)):
        err=err+((np.log(p[i]+1)-np.log(a[i]+1))**2)    
        total_error=(np.sqrt(err/len(p)))
    return total_error


# code taken from,

# https://stackoverflow.com/questions/238260/how-to-calculate-the-bounding-box-for-a-given-lat-lng-location/238558#238558
# by Federico A. Ramponi (https://stackoverflow.com/users/18770/federico-a-ramponi)
# This snippet of code basically takes a set of latitude and longitude coverts it into radius(distance) due to \
# speroidical shape of earth \
# and returns 4 coordinates which surround the set of latitude and longitude as a box.


# degrees to radians
def deg2rad(degrees):
    return math.pi*degrees/180.0

# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi

    # Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))

def final_fun_2(air_visit_data, air_store_info, hpg_store_info, date_info, store_id_relation):

    
    bounding_box_lat=[]
    bounding_box_lon=[]
    for i in range(len(air_store_info)):
        bounding_box_lat.append(air_store_info['latitude'][i])
        bounding_box_lon.append(air_store_info['longitude'][i])
        
    neighbour=[]
    lat_1=[]
    lon_1=[]
    lat_2=[]
    lon_2=[]
    for i in range(len(air_store_info)):    
        lat1, lon1, lat2, lon2=boundingBox(bounding_box_lat[i],bounding_box_lon[i],1.5)
        lat_1.append(lat1)
        lon_1.append(lon1)
        lat_2.append(lat2)
        lon_2.append(lon2)  
        
    for i in range(len(air_store_info)):
        count=0
        for j in range(len(air_store_info)):        
            if bounding_box_lat[j]>lat_1[i] and bounding_box_lat[j]<lat_2[i] and bounding_box_lon[j]>lon_1[i] and bounding_box_lon[j]<lon_2[i]:
                count=count+1
        neighbour.append(count-1) 
        
    air_store_info['nearest_neighbour']=neighbour
    
    air_store_info=air_store_info.rename(columns={"air_genre_name":"genre_name","air_area_name":"area_name"})
    hpg_store_info=hpg_store_info.rename(columns={"hpg_genre_name":"genre_name","hpg_area_name":"area_name"})
    
    date_info=date_info.rename(columns={"calendar_date":"visit_date"})
    
    total_data=pd.merge(air_visit_data,date_info,how='left',on=['visit_date'])
    total_data=pd.merge(total_data,store_id_relation,how='left',on= ['air_store_id'])
    total_data=pd.merge(total_data,air_store_info,how='left',on= ['air_store_id'])
    total_data=pd.merge(total_data,hpg_store_info,how='left',on= ['hpg_store_id','latitude','longitude','genre_name','area_name'])
    
    total_data=total_data.drop(columns=['hpg_store_id'])
    
    total_data['day_of_week']=total_data['day_of_week'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],[1,2,3,4,5,6,7])
    
    total_data['latlong']=(0.5*total_data['latitude'])+(2*total_data['longitude'])
    
    #day_of_week=total_data['day_of_week']
    holiday = total_data.apply(lambda x : x.holiday_flg==1, axis=1)
    holiday_prev = total_data.apply(lambda x : x.day_of_week and x.holiday_flg+1==1, axis=1)
    total_data.loc[holiday,'day_of_week'] = 6
    total_data.loc[holiday_prev,'day_of_week'] = 5         
            
    
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    le1.fit(total_data['air_store_id'].unique())
    le2.fit(total_data['area_name'].unique())
    le3.fit(total_data['genre_name'].unique())
    
    pickle.dump(le1,open('LaEn_air_store_id.pkl','wb'))
    pickle.dump(le2,open('LaEn_area_name.pkl','wb'))
    pickle.dump(le3,open('LaEn_genre_name.pkl','wb'))


    total_data['label_encoded_id']=le1.fit_transform(total_data['air_store_id'])
    total_data['label_encoded_area']=le2.fit_transform(total_data['area_name'])
    total_data['label_encoded_genre']=le3.fit_transform(total_data['genre_name'])
    
    total_data['visit_date']=pd.to_datetime(total_data['visit_date'])
    total_data['visit_year']=total_data['visit_date'].dt.year
    total_data['visit_month']=total_data['visit_date'].dt.month
    total_data['visit_date']=total_data['visit_date'].dt.date
    
    total_data = total_data.drop(columns=["air_store_id","genre_name","area_name","visit_date"],axis=1)
    
    total_data = total_data[['visit_year','visit_month','day_of_week','holiday_flg','latitude','longitude','latlong','label_encoded_id','label_encoded_area','label_encoded_genre','nearest_neighbour','visitors']]
    
    total_data.to_csv("new_train_data.csv",index=False)
    
    #total_data=total_data[['visit_year','visit_month','day_of_week','holiday_flg','latitude','longitude','latlong','label_encoded_id','label_encoded_area','label_encoded_genre','nearest_neighbour','visitors']]
    
    Y = total_data["visitors"].values
    X = total_data.drop(columns=["visitors"],axis=1)
    
    # train test split (80:20)
    X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size=0.20, random_state=42)   

    model = lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=31,max_depth=-1,learning_rate=0.1, 
                        n_estimators=100, colsample_bytree=1.0, random_state=None, 
                        n_jobs=-1)
    
    model.fit(X_train,y_train)
    tr_preds=model.predict(X_train)
    cv_preds=model.predict(X_cv)
        
    train_error=0
    cv_error=0  
    
    train_error=root_mean_squared_logarithmic_error(cv_preds,y_cv) 
    cv_error=root_mean_squared_logarithmic_error(tr_preds,y_train)
    
    pickle.dump(model,open('lightgbm.pkl','wb'))

    
    return train_error,cv_error


train_error, cv_error = final_fun_2(air_visit_data, air_store_info, hpg_store_info, date_info, store_id_relation)
print("Train Score: ",train_error)
print("CV Score: ",cv_error)        