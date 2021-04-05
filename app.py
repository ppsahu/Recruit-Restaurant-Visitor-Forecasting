# A simple script to do MAthematical calaculation
from pywebio.input import *
from pywebio.output import *

# importing all the required libraries
import numpy as np
import pandas as pd
from datetime import datetime
import time, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle
import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import math
from tqdm import tqdm
from pywebio import start_server

# Reading all the files
air_visit_data = pd.read_csv('air_visit_data.csv')
air_store_info = pd.read_csv('air_store_info.csv')
air_reserve = pd.read_csv('air_reserve.csv')
hpg_store_info = pd.read_csv('hpg_store_info.csv')
hpg_reserve = pd.read_csv('hpg_reserve.csv')
date_info = pd.read_csv('date_info.csv')
store_id_relation = pd.read_csv('store_id_relation.csv')
sample_submission = pd.read_csv('sample_submission.csv')

label1 = joblib.load('LaEn_air_store_id.pkl')
label2 = joblib.load('LaEn_area_name.pkl')
label3 = joblib.load('LaEn_genre_name.pkl')
model = joblib.load('lightgbm.pkl')

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


def final_fun_1(restaurant_id,date_info,store_id_relation,air_store_info,hpg_store_info):
    
    sample_submission = []
    
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
    
    # extracting the id and date from "id" column
    a = restaurant_id[:20]
    b = restaurant_id[21:]
    c = pd.to_datetime(b)
    
    d = c.dayofweek
    e = c.year
    f = c.month

    s=[]
    s=[[a,c,d,e,f]]
    sample_submission = pd.DataFrame(s,columns=['air_store_id','visit_date','visit_dow','visit_year','visit_month'])

    
    date_info = date_info.rename(columns={"calendar_date":"visit_date"})
    date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
    
    test_data=pd.merge(sample_submission,date_info,how='left',on=['visit_date'])
    test_data=pd.merge(test_data,store_id_relation,how='left',on= ['air_store_id'])
    test_data=pd.merge(test_data,air_store_info,how='left',on= ['air_store_id'])
    test_data=pd.merge(test_data,hpg_store_info,how='left',on= ['hpg_store_id','latitude','longitude','genre_name','area_name'])

    test_data['day_of_week']=test_data['day_of_week'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],[1,2,3,4,5,6,7])
    
    test_data['latlong']=(0.5*test_data['latitude'])+(2*test_data['longitude'])

    test_data['label_encoded_id']=label1.transform(test_data['air_store_id'])
    test_data['label_encoded_area']=label2.transform(test_data['area_name'])
    test_data['label_encoded_genre']=label3.transform(test_data['genre_name'])

    test_data=test_data.drop(columns=['air_store_id','visit_date','genre_name','area_name','visit_dow','hpg_store_id'])

    test_data = test_data[['visit_year','visit_month','day_of_week','holiday_flg','latitude','longitude','latlong','label_encoded_id','label_encoded_area','label_encoded_genre','nearest_neighbour']]

    test_data = test_data[['visit_year','visit_month','day_of_week','holiday_flg','latitude','longitude','latlong','label_encoded_id','label_encoded_area','label_encoded_genre','nearest_neighbour']]

    predictions = model.predict(test_data)

    put_text("The predicted number for visitors for  restaurant ",restaurant_id," and date is: ",predictions)

    #return predictions
    

if __name__ == '__main__':

    restaurant_id = radio("Choose one restaurant and date", options=[
        "air_00a91d42b08b08d9_2017-04-23",
        "air_00a91d42b08b08d9_2017-04-24",
        "air_00a91d42b08b08d9_2017-04-25",
        "air_00a91d42b08b08d9_2017-04-26",
        "air_00a91d42b08b08d9_2017-04-27",
        "air_0164b9927d20bcc3_2017-04-28",
        "air_0241aa3964b7f861_2017-04-29",
        "air_0328696196e46f18_2017-04-30",
        "air_034a3d5b40d5b1b1_2017-05-01",
        "air_036d4f1ee7285390_2017-05-02",
        "air_0382c794b73b51ad_2017-05-03",
        "air_03963426c9312048_2017-05-04",
        "air_04341b588bde96cd_2017-05-05",
        "air_049f6d5b402a31b2_2017-05-06",
        "air_04cae7c1bc9b2a0b_2017-05-07",
        "air_0585011fa179bcce_2017-05-08",
        "air_05c325d315cc17f5_2017-05-09",
        "air_0647f17b4dc041c8_2017-05-10",
        "air_064e203265ee5753_2017-05-11",
        "air_066f0221b8a4d533_2017-05-12",
        "air_06f95ac5c33aca10_2017-05-13",
        "air_0728814bd98f7367_2017-05-14",
        "air_0768ab3910f7967f_2017-05-15",
        "air_07b314d83059c4d2_2017-05-16",
        "air_07bb665f9cdfbdfb_2017-05-17",
        "air_082908692355165e_2017-05-18"
        ])
    final_fun_1(restaurant_id,date_info,store_id_relation,air_store_info,hpg_store_info)
  
    
