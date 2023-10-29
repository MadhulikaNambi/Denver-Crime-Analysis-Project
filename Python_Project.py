#!/usr/bin/env python
# coding: utf-8

# # BUDT 704 Project Report

# ## Project Title: Law and Order: Denver Edition 

# # Introduction

# When we think of Denver, its scenic beauty is the first thing that comes to mind. Due to a recent spike in crimes, Denver has become associated with crimes in peoples' minds whenever they think of the city. The Denver Police Department and the city's residents both want this perception to change. To help them protect their citizens and uphold the sense of security that Denver exudes to its residents, visitors, and onlookers, they have hired Denver Mifflin, a consultancy firm based in the city. <br>
# 
# We at Denver Mifflin want Denver residents to enjoy their city's picturesque streets without having to worry about becoming a victim of crime. The government of Denver has provided us with a dataset that has recorded crimes since 2017. Denver Mifflin, a group of seven consultants, works to restore safety to Denver by analyzing crime statistics and giving the Denver Police Department useful information they can use to combat crime in the future.<br>
# 
# At Denver Mifflin, we are confident that at the conclusion of our analysis, we will be able to assist the police as well as the kind residents of Denver in altering people's perceptions of the city and restoring Denver to its former status as one of the safest and most beautiful cities in the nation.
# 
# 

# ### Research Questions

# **Q1. Analyze the intervals between each occurrence of a given crime to identify the types of crimes that occur regularly.**<br>
#     
# **End Goal**: By determining crimes that occur almost on a day to day basis, we can help the Denver Police Department identify the types of crimes they need to address on priority. By reducing the instances of such crimes, the city of Denver will see a downward trend in the number of crimes committed each day. This will in turn bring down the yearly crime rate significantly.<br>
#    
# **Q2. By analyzing the occurrence of various crimes, we strive to identify patterns based on understanding which crimes typically occur on which days in order to provide actionable insights to the Denver Police Department.**<br>
# 
# **End Goal**: By determining the trends of crimes and identifying the days on which each crime typically takes place, we would like to provide recommendations to the Denver Police Department on how to handle these crimes and take precautionary measures.<br>
#     
# **Q3.  Derive insights on how the nature of the crime affects the time taken to report the crime.** <br>
#     
# **End Goal**: We would like to analyze the time taken for specific crimes to be reported and incase of any delays, understand the underlying cause behind this delay in reporting time. Based on this, we can recommend the Denver Police Department to create public awareness campaigns and provide necessary protocols to encourage victims to report such crimes promptly.<br>
# 
# **Q4. Predicting the likelihood of a person being in an unsafe neighborhood in Denver based on the individual's coordinates.**<br>
# 
# **End Goal**: We are attempting to map out potentially unsafe areas across Denver, classify them by likelihood of crime and send out basic safety measures based on an individual's co-ordinates. This will allow people to either steer clear of the shady areas of Denver or be cautious while in those areas. Achieving this will be a win-win scenario as the police will have the access to the same algorithm, so when a person in distress calls a simple check on their location will allow the police to prepare for kind of response they need to engage in, trying to diffuse the situation.
# 

# ### Data Sets

# The first dataset aquired is the **crime** data set from  https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-crime. The data is downloaded directly as a csv file.
# 
# The second data set that is used is the **traffic accident** dataset from  https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-crime. The data is downloaded directly as a csv file. 
# 
# **Permission for use**:
# 
# According to the Open Data License present on the website, "You are free to copy, distribute, transmit and adapt the data in this catalog under an open license. For further information refer to https://www.denvergov.org/opendata/termsofuse".
# <br>
# 
# The crime data set currently has 382831 rows and 20 columns. <br>
# The traffic accident data set is only going to be used to pull the lattitudes and longitudes of various parts of denver that will be used to build a validation dataset for the model in the later part of the project.

# ### Choice for Heavier Grading on Data Processing or Data Analysis

# We choose to be heavily graded on __Data Analysis__

# __Resons for Choice:__ <br>
# The data set was directly downloaded from the Denver Government Website and as a result there was not much pre-processing work that had to be done. In short, the data didn't require extensive processing.<br>
# We predominantly wanted to focus on identifying patterns and deriving actionable insights from data, and visually depicting our findings for easy understanding.Our data had multiple parameters with respect to loactions and time which we have used extensively in our analysis.

# # Data Processing 

# We will now try to import various packages for our analysis. We will be using packages for various operations in this notebook.
# 
# - numpy - for performing array operations
# - pandas - to perform operations on dataframes
# - plotly - to map our data points to a map 
# - sklearn - for predictive analytics
# - seaborn, matplotlib, plotly - for visualizations
# - folium - visualizations for geo-location points

# In[2]:


# Loading librares to be used 
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


master_data = pd.read_csv("crime_data.csv",encoding="latin1") # reading the data into a dataframe 


# ## Initial Basic Analysis of Data. 

# In[5]:


# We try undertand the data set in detail
master_data.info()


# We find that the **last occurence date is NULL for almost 50% of the data**. This could be probably because most crimes in our dataset happen once, and as a result they would not be considered as recurring offences. The data that is missing in the **INCIDENT_ADDRESS,GEO_LAT** and **GEO_LON** columns are because they are sensitive crimes like Sexual Assault for example, and information about those crimes are not disclosed for privacy and safety concerns. 

# The column **'OFFENSE_TYPE_ID'** has a lot of abbreviated words, for example, **'assault'** is represented as **'aslt'**. To maintain uniformity, we replace all the abbreviated words with their full forms.<br>
# To acheive this task we first build a dicitonary with the key as abbreviated words and value as that words' full form. With the help of the dictinary, we replace the abbreviations in our main data frame. The reason we have taken up the dictionary approach is because of the comprehensive list of changes that need to be made, in order to make it more readable.<br>
# 
# We further make modifications to few of our columns using `regex`. These changes are done by replacing current garbage characters with proper characters that are necessary for our analysis.<br> 
# Our end objective is to display our analysis in charts. In order to plot them, we made use of `Title`, to make it appear creative.<br>
# We have columns which contain datetime information. For better extraction of this information, we have transformed them into a certain format for easy analysis. We have also fetched **Year & Month** from the column which has date information using `datetime`. <br>
# 
# We perform column opertions to derive new columns which will later be used during our analyis. For example, we are trying to calculate the difference between two time intervals, in this case, **date**.

# In[278]:


# Replacing based on key value pair format of dictionary
replacement_dict = {'agg': 'aggravated', 'aslt' : 'assault', 'vin' : 'Vehicle Identification Number', 'poss' : 'possession', 'dv' :'domestic violence', 'burg' : 'burglary', 'busn' : 'business', 'resd' :'residential', '-w-' : 'with', 'mtr' : 'motor', 'veh' : 'vehicle', 'mfr': 'manufacture', 'deriv' : 'derivative','possess' :'possession', 'synth': 'synthetic', 'pos': 'possession','dev':'development','viol': 'violation', 'occ':'occupied','stln' : 'stolen', 'ftd':'Financial Transaction Device', 'unauth':'unauthorized','const': 'constant','bldg':'building', 'sex':'sexual','off': 'offender', 'asslt': 'assault','pot':'potential', 'rpt' : 'report','weap':'weapon','nsf':'non sufficient funds','inst': 'instrument','all-other-crimes':'Felonies and misdemeanor','other-crimes-against-persons':'Crimes with malicious intent'}
master_data['OFFENSE_TYPE_ID'].replace(replacement_dict,regex=True,inplace=True) # Replacing values as 


# In[ ]:


#  Removing all the '-' from the columns
master_data['OFFENSE_TYPE_ID'] = master_data['OFFENSE_TYPE_ID'].str.replace('-',' ',regex = True)
master_data['OFFENSE_CATEGORY_ID'] = master_data['OFFENSE_CATEGORY_ID'].str.replace('-',' ',regex = True)
master_data['NEIGHBORHOOD_ID'] = master_data['NEIGHBORHOOD_ID'].str.replace('-',' ',regex = True)

# Capitalizing all the first words of the Column values
master_data['OFFENSE_TYPE_ID'] = master_data['OFFENSE_TYPE_ID'].str.title()
master_data['OFFENSE_CATEGORY_ID'] = master_data['OFFENSE_CATEGORY_ID'].str.title()
master_data['NEIGHBORHOOD_ID'] = master_data['NEIGHBORHOOD_ID'].str.title()

master_data[['FIRST_OCCURRENCE_DATE','FIRST_OCCURRENCE_TIME']]= master_data['FIRST_OCCURRENCE_DATE'].str.split(' ',1,expand=True) #Splitting the data
master_data[['REPORTED_DATE','REPORTED_TIME']]= master_data['REPORTED_DATE'].str.split(' ',1,expand=True) #Splitting the data
master_data=master_data[['INCIDENT_ID','OFFENSE_ID','OFFENSE_CODE','OFFENSE_CODE_EXTENSION','OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE','FIRST_OCCURRENCE_TIME','LAST_OCCURRENCE_DATE','REPORTED_DATE','REPORTED_TIME','INCIDENT_ADDRESS','GEO_X','GEO_Y','GEO_LON','GEO_LAT','DISTRICT_ID','PRECINCT_ID','NEIGHBORHOOD_ID','IS_CRIME','IS_TRAFFIC']]
master_data.drop(columns=['LAST_OCCURRENCE_DATE'],inplace=True) # Dropping this column as it has no much use towards the analysis


# In[193]:


#Extracting the year and month and storing them in seprate columns
master_data['CRIME_OCCURRENCE_YEAR'] = pd.DatetimeIndex(master_data['FIRST_OCCURRENCE_DATE']).year
master_data['CRIME_OCCURRENCE_MONTH'] = pd.DatetimeIndex(master_data['FIRST_OCCURRENCE_DATE']).month


# In[194]:


#Calculating time difference between the reported date and the first occurence date
master_data.loc[:, 'FIRST_OCCURRENCE_DATE'] = pd.to_datetime(master_data.loc[:, 'FIRST_OCCURRENCE_DATE'])
master_data.loc[:, 'REPORTED_DATE'] = pd.to_datetime(master_data.loc[:, 'REPORTED_DATE'])

master_data['TIME_DIFFERENCE'] = master_data['REPORTED_DATE'] - master_data['FIRST_OCCURRENCE_DATE']
master_data['DAYS_DIFFERENCE'] = pd.to_timedelta(master_data.TIME_DIFFERENCE, errors='coerce').dt.days #building the time delta variable


# In[195]:


master_data


#  

# In[269]:


accident_data = pd.read_csv('traffic_accidents.csv',usecols = ['geo_lon','geo_lat','neighborhood_id']) # reading the data set for for building validation data set


# In[270]:


#Cleaning the data to bring uniformity
accident_data['neighborhood_id'] = accident_data['neighborhood_id'].str.replace('-',' ',regex = True)
accident_data['neighborhood_id'] = accident_data['neighborhood_id'].str.title()


# # DATA ANALYSIS
# 
# We try to identify patterns and find answers for certain questions that we believe will help in deriving actionable insights, using which we can communicate our findings and provide recommendations to the Denver Police Department.

# **Q1. Analyze the intervals between each occurrence of a given crime to identify the types of crimes that occur regularly.**<br>

# __Introduction:__<br>
# 
# Denver Mifflin wants to start off by analysing which crimes occur the most frequently. By doing so, we can get a clear idea as to which crimes are more prevalant in Denver. Upon finding the most occuring crimes in the city of Denver, we take a step ahead and visualise the areas where such incidents have been reported. We will try to apply some aggregation to our dataset and visualise some meaningful outputs from it.
# 

# In[272]:


data_for_analysis1 = master_data.copy() #Making a copy of our master data set


# To beging our analysis we sort the values by date because this will enable us to calculate the difference between two consecutive rows which is essential for our analysis. 

# In[273]:


data_for_analysis1.sort_values(by=['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE'],inplace=True) #Sorting the values by date 

#Calculating the difference between two consecutive rows in the dataframe
data_for_analysis1['Date Diff']=data_for_analysis1.groupby(['OFFENSE_CATEGORY_ID'])['FIRST_OCCURRENCE_DATE'].diff()

#Grouping the values so the average is taken for each offence category
data_as_per_offense_category= data_for_analysis1.groupby('OFFENSE_CATEGORY_ID')['Date Diff'].mean()

data_as_per_offense_category = data_as_per_offense_category.to_frame() #Converting the groupby object to a data frame to perform analysis


# In[274]:


data_as_per_offense_category


# In[275]:


# Calculating the Hour difference 
data_as_per_offense_category['Hour diff']= data_as_per_offense_category.apply(lambda x: (x['Date Diff']/dt.timedelta(hours=1)), axis=1)
  

data_as_per_offense_category.drop('Date Diff',axis=1,inplace=True)

data_as_per_offense_category.sort_values(by = ['Hour diff'], inplace = True)
print(data_as_per_offense_category)


# ## Analysis Findings
# As seen above 'Theft from Motor Vehicles' crimes are the frequently happening crimes. We try to further investigate and address this particular type of crime which we believe will help in bringing down crime rate in Denver significantly.

# In[276]:


theft_from_motor_vehicle = master_data[(master_data['OFFENSE_CATEGORY_ID'] == 'Theft From Motor Vehicle')].copy() #Building a data frame of only Theft from Motor vehicle data

theft_from_motor_vehicle.dropna(axis=0,how='any',subset=['GEO_LAT','GEO_LON','OFFENSE_CATEGORY_ID'],inplace=True) #Dropping any null values


# To begin, We design an interactive map that identifies areas where 'Theft from motor vehicle' crimes predominantly occur.

# In[277]:


# We want the map to open at the city of Denver so we set the mean Latitutde and Longitutde of the data.
map_center_lon=theft_from_motor_vehicle['GEO_LON'].mean() #Calculating the mean of he Longitutde
map_center_lat=theft_from_motor_vehicle['GEO_LAT'].mean() #Calculating the mean of he Latitiude
main_map=folium.Map([map_center_lat,map_center_lon],zoom_start=12) # Setting an up and empty map

theft_map=plugins.MarkerCluster().add_to(main_map)
for latttitude,longitude,offence_category_id in zip(theft_from_motor_vehicle.GEO_LAT,theft_from_motor_vehicle.GEO_LON,theft_from_motor_vehicle.OFFENSE_CATEGORY_ID): # Taking a loop to browse through all the rows of the data frame
    folium.Marker(location=[latttitude,longitude],icon=None,popup=offence_category_id).add_to(theft_map)
main_map.add_child(theft_map) # adding a child node to the graph

main_map # Displayign the map


# **Observations**:<br>
# We observe that the 'Theft from Motor Vehicles' crime is spread out through out the map. This article on the internet ('https://www.westword.com/news/denver-is-one-big-car-theft-hot-spot-15306202') states that Denver leads the nation in the Motor Vehicles Theft. But within Denver, we can observe that criminals are preferring to steal from the vehicle rather than the vehicle itself.<br>
# 
# **Inference**:<br>
# According to this news article ('https://kdvr.com/news/data/denver-homeless-population-point-in-time-count-2022/') homelessness has been on the rise in Denver and has reached a 14-year record in 2022. It states that the number of homeless people has reached approximately 30,000. Correlating this article with the following news article ('https://www.9news.com/article/news/crime/reasons-behind-colorado-car-thefts/73-603b4e70-6329-4600-8238-769c0e73d5c8') which tells the case if a Auto Car thief 'Illya Culpepper', we can make an inference that homeless people break into cars just because they need some place to sleep and endure the harsh cold of Denver as we can see from ('https://en.wikipedia.org/wiki/Geography_of_Denver'). While they are in the cars they pilage the car for some cash, cards or other valuables.<br>
# 
# **Recommendations**:<br>
# Our suggestions to the Denver Police Department would be:<br>
# 1. Increase patrolling near secluded parking-lot <br>
# 2. Make No-Parking zones/ Limited-Time Parking zones near places most prone to these crimes<br>
# 
# We would also advise the Police department to issue the following notice to the genral public:<br>
# 1. Always park in well-lit areas<br>
# 2. Keep all valuables in the trunk or hidden from view<br>
# 3. Always keep your car doors locked, when driving or parked<br>
# 4. If you are approached, do not roll down the windows or open a door<br>

# **Q2. By analyzing the occurrence of various crimes, identify patterns based on understanding which crimes typically occur on which days in order to provide actionable insights to the Denver Police Department.**

# ## Introduction
# Denver Mifflin wants to concentrate on offering suggestions to the Denver Police Department regarding how they might combat crime in the city of Denver. We start by identifying the day of the week on which crimes occur. We look for patterns in our analysis and try to find out if crimes that fall under a single category occur more on a specific day of the week or a couple of days of the week than others. We attempt to offer the Denver Police Department useful insights and suggestions based on our analysis on how they may go about reducing crime in the city.
# 

# In[196]:


# Sorting the data based on each offense catrgory with the earliest date
sorted_df = master_data.sort_values(['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE'],ascending=True).groupby('OFFENSE_CATEGORY_ID').head(50000000)


# In[197]:


# Subsetting our columns 
analysis_df = sorted_df[['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE']]


# In[198]:


# Subsetting our columns 
analysis_df = sorted_df[['OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE']]


# In[199]:


get_ipython().run_cell_magic('capture', '--no-stdout', "analysis_df['Difference'] = analysis_df.groupby('OFFENSE_CATEGORY_ID')['FIRST_OCCURRENCE_DATE'].diff()")


# In[200]:


get_ipython().run_cell_magic('capture', '--no-stdout', "# Calculating difference between each incident between each type of crime\nanalysis_df['Difference'] = analysis_df.groupby('OFFENSE_CATEGORY_ID')['FIRST_OCCURRENCE_DATE'].diff()\n# While the dataframe is re-ordered, we need to reset index\nanalysis_df.reset_index(drop=True, inplace=True)\n# Converting time based column to string for easy manipulation\nanalysis_df['Difference'] = analysis_df['Difference'].astype(str)\n# Using regex to remove 'days' in order to do quantitative analysis\nanalysis_df['Difference'] = analysis_df['Difference'].str.replace('days',' ',regex = True)\n# Filling values with zero where there is no data\nanalysis_df['Difference'].fillna(0, inplace=True)\nanalysis_df['Difference'] = analysis_df['Difference'].str.replace('NaT','0',regex = True)\n# Converting column to integere for quantitative analysis\nanalysis_df['Difference'] = analysis_df['Difference'].astype(int)\n# Fetching day name from the date column which we have in the data\nanalysis_df['FIRST_OCCURRENCE_DAY_NAME'] = analysis_df['FIRST_OCCURRENCE_DATE'].dt.day_name()")


# In[201]:


analysis_df.head(10)


# In[202]:


pivot_df = analysis_df.pivot_table(index=['FIRST_OCCURRENCE_DAY_NAME'],
               columns=['OFFENSE_CATEGORY_ID'], 
               aggfunc={'OFFENSE_CATEGORY_ID': 'count'})


# In[203]:


pivot_df.reset_index()
pivot_df


# ### Analysis Findings
# We identify a trend in the days of the week that crimes occur after doing our analysis. Most crimes fall into one of two categories: those that happen mostly during the week and those that happen mostly on the weekends. We try to address these crimes and determine whether the crimes are connected and whether if there are any general steps that can be taken to stop them. 
# 

# ## Crimes on Weekends

# In[279]:


pivot_df2 = pivot_df.loc[:, 'OFFENSE_CATEGORY_ID']
pivot_df2[['Aggravated Assault','Murder','Public Disorder','Robbery','Sexual Assault']]


# 
# The following crimes primarily occur on weekends, as can be seen from the data frame above:
# - Aggravated Assault
# - Murder
# - Public Disorder
# - Robbery
# - Sexual Assault
# 
# 

# **Let us look at the pattern of 'Aggravated Assault' crimes for example**.

# In[280]:


chart_df_1 = analysis_df[analysis_df.OFFENSE_CATEGORY_ID == 'Aggravated Assault']

plt.figure(figsize=(15,8))
ax = sns.countplot(x=chart_df_1["FIRST_OCCURRENCE_DAY_NAME"],
             order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],
             palette=['#FF0000','#808080','#808080','#808080','#808080','#808080','#FF0000'])
ax.set(xlabel='Day of the week', ylabel='Number of Crimes')
ax.set_title('Aggravated Assault Crimes by Day of Week')


# ### Observation
# 
# We provide evidence to support our hypothesis that weekends are when the majority of 'Aggravated Assault' crimes occur. We also observe that the majority of crimes that take place on the weekends follow the trend, from the data frame above. As mentioned in the journal 'Patterns in Criminal Aggravated Assault' published by Northwestern University (https://scholarlycommons.law.northwestern.edu/cgi/viewcontent.cgi?article=5261&context=jclc), during week days, interaction among people is limited by their work, and there is less leisure time than on the weekends. For this reason, the majority of the acts of aggravated assault would occur between 6:00 p.m. Friday and 6:00 a.m. Monday. Criminal aggravated assaults may occur in a wide variety of places, but it was hypothesized that place of occurrence would be related to the season of the year, victim-offender relationship, and sex status. It was noted that the largest number of acts would occur on public streets, rather than in taverns or bars, residences, or other places. We see that most of the crimes that are committed on weekends are similar types of offenses, i.e., they all predominantly occur in public and are impulsive crimes. These crimes happen in the heat of the moment and often are not planned. We try to understand the major cause of such crimes.
# 
# ### Inference
# Every year, millions of crimes are committed by offenders under the influence of alcohol. These offenses include typical alcohol-involved crimes like driving while intoxicated, but also public disorder, vandalism, theft, robbery, domestic violence, assault, rape, and murder. According to a report publish by the Manhattan Institute, "Fixing Drinking Problems: Evidence and Strategies for Alcohol Control as Crime Control" (https://www.manhattan-institute.org/evidence-and-strategies-for-alcohol-control-as-crime-control) remarkably strong base of evidence confirms more generally that alcohol consumption and availability cause many crimes. The report states that greater consumption translates into a nearly 6% greater arrest rate, driven by robberies, assaults, alcohol-related offenses, and nuisance crimes. It further states that violence is disproportionately common in and around venues that serve alcohol. Data from the National Incident-Based Reporting System, for example, indicate that murders and assaults are two to three times more likely to occur in bars, relative to other places. These results suggest that beefing up security at and surrounding drinking establishments will considerably reduce public crime, particularly on weekends. 
# 
# ### Recommendations
# By keeping people from committing crimes while intoxicated, the Denver Police Department may drastically reduce weekend crime. These kinds of crimes can be quickly addressed, which will aid in reducing crime overall in Denver, particularly on the weekends. Some of our recommendations for the Denver Police Department to reduce crime on weekends are:
# 
# - Beefing up security around all establishments that serve alcohol.
# - Making sure adequate lights are installed in all locations to deter criminals from making a move.
# - Increased patrolling throughout the city, particularly at night.
# - Ensuring establishments that serve alcohol do not serve alcohol after the permitted hours.
# - Have an active helpline number that will be prepared to immediately assist people who are in danger.
# - Have police personnel in each city zone to constantly monitor CCTV camera footage and coordinate with patrol squads in the location to prevent crimes before they can actually happen.
# 
# We at Denver Mifflin think that by following the aforementioned advice, the city of Denver's weekend crime rate will dramatically decrease. We offer the aforementioned suggestions because we think they can be put into practice right away using the resources at hand.
# 

# ## Crimes on Weekdays

# In[323]:


pivot_df2[['Burglary','White Collar Crime','Larceny','Theft From Motor Vehicle']]


# The following crimes primarily occur on weekdays, as can be seen from the above data frame:
# - Burglary
# - White Collar Crime
# - Larceny
# - Theft From Motor Vehicle

# **Let us look at the pattern of 'Burglary' crimes for example.**

# In[309]:


chart_df_2 = analysis_df[analysis_df.OFFENSE_CATEGORY_ID == 'Burglary']
plt.figure(figsize=(15,8))
ax = sns.countplot(x=chart_df_2["FIRST_OCCURRENCE_DAY_NAME"],
             order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],
             palette=['#808080','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#808080'])
ax.set(xlabel='Day of the week', ylabel='Number of Crimes')
ax.set_title('Burglaries by Day of Week')


# We provide evidence to support our hypothesis that weekdays are when the majority of crimes that fit into this category occur. We also observe that the majority of crimes that take place on the weekdays follow the trend, from the data frame above. We make an effort to address crimes that the Denver Police Department ought to handle first. The following offence is chosen for this purpose as we believe that burglaries occur on a daily basis and that the Denver Police Department have sufficient resources to combat these crimes:

# ### Burglary
# Burglaries predominantly take place during weekdays. We try to understand why they follow such a pattern. Research actually suggests that break-ins are most likely to happen during the day. Burglars are most likely to enter homes on weekdays between 10 a.m. and 11 a.m. or from 1 p.m. to 3 p.m., according to the U.S. Department of Justice. One of the major reasons for this is because most people are away for work or school during weekdays and this provides burglars the perfect window to commit burglaries. According to the article "Delving into the mind of a burglar" published by ADT Security (https://www.adtsecurity.com.au/blog/the-mind-of-a-burglar/), two-thirds of break-ins are through open windows or doors, which is great news for the opportunistic burglar who simply wanders around properties checking for unlocked or partially open access points. They are also wary of homes with a visible security system, and may move straight on to a neighbouring property if one is detected. To deter burglars and reduce burglaries, we provide the following recommendations:
# 
# - The Denver Police Department can come up with the concept of forming community groups in each area. These community groups will consist of representatives chosen by the community members, who'll continuously work for the welfare and protection of the community. Having a police officer outside every house is impractical and the Denver Police Department would not only help prevent crimes significantly by forming such groups but will also promote community growth and welfare.
# - The Denver Police Department should educate communites on the importance of home security cameras and must provide financial aid for community groups to install home security systems. Stats show that installing a home security system deters burglars and this would significantly reduce the number of burglaries.
# 
# By following the above recommendations, we believe the number of burglaries will significantly come down in the city of Denver.
# 
# From all the crimes the typically occur during weekdays, we at Denver Mifflin recommend the Denver Police Department to address the above crimes on priority as they are crimes that are happening on a day to day basis but can easily be prevented. By addressing all the crimes stated above, we believe the overall crime rate in Denver will significantly come down.

# **Q3. Derive insights on how the nature of the crime affects the time taken to report the crime.** <br>

# ## Introduction
# Denver Mifflin wants to additionally analyze the time taken for specific crimes to be reported and determine the underlying cause behind any potential delay in reporting time. We start by determining the average reporting time for each crime category. From this, we can identify the crimes that have the longest reporting time.
# 
# 
# 

# In[311]:


data_for_analysis = master_data.copy()


# In[312]:


# Performing a group by function to categorize data by the average amount of time taken 
values_as_per_group = data_for_analysis.groupby(by=data_for_analysis['OFFENSE_CATEGORY_ID'])['DAYS_DIFFERENCE'].agg('mean').sort_values(ascending=False)


# In[313]:


values_as_per_group


# Looking at the above demographic we understand that it takes the longest to report Sexual Assault types of crimes.<br> 
# We perform a futher analysis to see what insights we gain

# In[314]:


sexual_assault_data = data_for_analysis[data_for_analysis['OFFENSE_CATEGORY_ID']=='Sexual Assault'].copy() #Segregating data that has sexual assualt as the Offense Category ID


# In order to predict a trend we try to try to see the number of cases of Sexual Assault Crimes each year with the help of the data frame below

# In[315]:


crime_year_values = sexual_assault_data["CRIME_OCCURRENCE_YEAR"].to_numpy() #Creating a array of the values so that they can be used for plotting

years,crime_counts=np.unique(crime_year_values,return_counts=True)# counting the unique values in the array 

sexual_assualt_reporting  = pd.DataFrame(data=np.column_stack((years,crime_counts)),columns=['Years','Crime Count'])

sexual_assualt_reporting


# As seen above, the number of cases per year reduced significantly from the year 2019 to 2020. **Could the COVID-19 pandemic have made an effect of the number of sexual assault cases being reported?** To gain further insights we try to understand what was the average reporting time of Sexual Assault cases per year.

# In[316]:


avg_sexual_assault_reporting_time = sexual_assault_data.groupby(by=sexual_assault_data['CRIME_OCCURRENCE_YEAR'])['DAYS_DIFFERENCE'].agg('mean').sort_values(ascending=False) # Calculating the average time report sexual assualt cases


# In[317]:


avg_sexual_assault_reporting_time_per_year = pd.DataFrame(data=np.column_stack((avg_sexual_assault_reporting_time.index,avg_sexual_assault_reporting_time.values)),columns=['Years','Avg Reporting Time']) # Creating a data frame of the result


# In[318]:


avg_sexual_assault_reporting_time_per_year


# In[319]:


sexual_assault_data.dropna(axis=0,how='any',subset=['GEO_LAT','GEO_LON'],inplace=True) # dropping the null values


# In[320]:


avg_reporting_time = dict(zip(avg_sexual_assault_reporting_time.index,avg_sexual_assault_reporting_time)) # building a dictionary of avg reporting time against the years


# In[321]:


sexual_assualt_reporting['Reporting Time (Days)'] = sexual_assualt_reporting['Years'].map(avg_reporting_time) # adding the values to the data frame


# In[307]:


sexual_assualt_reporting['Reporting Time (Days)'] = round(sexual_assualt_reporting['Reporting Time (Days)'])# rounding the values 


# In[322]:


sexual_assualt_reporting


# In[308]:


fig = px.bar(sexual_assualt_reporting, x="Years", y=["Crime Count", "Reporting Time (Days)"], title="Number of Sexual Assault Cases vs Average Reporting Time",text_auto=True,height = 500)
fig.update_layout(legend_title = '')
fig.show()


# ## Observation & Inference
# Looking at the above graph we understand that it takes the longest to report Sexual Assault crimes. The intensity of the trauma inflicted by a crime affects the time taken by the victim to report the crime. As observed in the graph, it takes the most time, i.e., nearly 50 days to report a sexual assault. According to https://www.brennancenter.org/our-work/analysis-opinion/sexual-assault-remains-dramatically-underreported, almost 80% sexual assault crimes are unreported, while 23% report the crime. Victims may not report the crime due to stigma associated with the crime, emotional barriers faced by the victim, external forces trying to suppress them or even unsuspected and known perpetrators.
# According to https://www.denverda.org/sexual-assault/ to The Denver government set up initiatives in their collaborative network as listed below:
# - Sexual Assault Response Team (SART)
# - Denver Anti-Trafficking Alliance (DATA)
# - Sexual Assault Interagency Council (SAIC)
# 
# 
# ## Recommendations
# Despite the government's efforts, these initiatives have not made a significant impact in reducing the reporting time. According to  https://www.brennancenter.org/our-work/analysis-opinion/sexual-assault-remains-dramatically-underreported, 20% victims are concerned about the backlash from society and 8% believe sexual assault is not significant enough to warrant reporting. To overcome this notion the Denver Government must consider educating people and spreading awareness about sexual violence along with incorporating secure channels for easier reporting of such crimes. 
# 
# ## Other Analysis Findings
# Another crime that is observed to take up to a month to get reported is a White Collar Crime. Many organizations have established their own regulatory agencies that keep an eye out for and protect businesses from such wrongdoings. These organizations first set up institutional trails and pass the judgement within their jurisdiction. Usually due to shame and self-blame for engaging in such fraudulent activities, most people don't report the crime. This could also be a reason for delay in reporting of such crimes. To prevent such crimes, there must be a major reform in the government where they enact laws and regulations that make it mandatory to report crimes that go undocumented and unnoticed to the public. 
# 
# Other crimes, such as larceny, motor vehicle theft, burglary and auto theft, take approximately the same amount of time to be reported. As per our expertise, we can say that such crimes are more likely to occur when people are away from their homes/vehicles for an extended period of time.  It is only after they return home, that they report the crime. The Government must deploy more resources towards patrolling and night watch to deter these crimes from happening.

# **Q4. Predicting the likelihood of a person being in an unsafe neighborhood in Denver based on the individual's coordinates.**<br>

# We begin our analysis by executing simple aggregations in order to find the percentage of crime by the neighbourhood ID, this part is essential for our model building and clustering later on.

# In[167]:


data_for_prediction = master_data.copy() # Creating a data frame for analysis


# In[168]:


#Groupig by the count of crimes per neighborhood
data_as_per_neighborhood = data_for_prediction.groupby(['NEIGHBORHOOD_ID'])['NEIGHBORHOOD_ID'].count() 
print(data_as_per_neighborhood)  


# In[169]:


data_as_per_neighborhood = data_as_per_neighborhood.to_frame() #Converting values to a data frame


# In[170]:


#Calculating the percentage of crimes as per the neigborhood this is calculated with respect to all the neighborhoods in the dataframe
data_as_per_neighborhood['NEIGHBORHOOD_ID'] = data_as_per_neighborhood['NEIGHBORHOOD_ID']/data_as_per_neighborhood['NEIGHBORHOOD_ID'].sum()*100


# In[171]:


data_as_per_neighborhood['NEIGHBORHOOD_ID']=round(data_as_per_neighborhood['NEIGHBORHOOD_ID'],2) #Rounding the values for better visualization


# In[172]:


data_as_per_neighborhood = data_as_per_neighborhood.rename(columns={'NEIGHBORHOOD_ID': 'Percentage of Crimes'}) #Renaming the column


# In[173]:


data_as_per_neighborhood.sort_values(by='Percentage of Crimes',ascending=False) #Seeing the regions with the most crime


# Once we obtain the percentage of crimes categorised by the neighborhood ID, we propose to keep them in seperate bins of our own creation. We are using bin names "High Crime", "Medium Crime","Low Crime", this gives rise to the question how do we justify the creation of the bins? The answer is simple yet brilliant, we want to let the user know about an attribute associated with their present location, this attribute is the crime percentage recorded in the area. When in situations of distress the simplest terms makes the most sense and elevates our sense of perceptibility and alertness. Keeping in mind with the psychological aspect of goal we will keep the usage of simple terms as a principle in the project. Once we rearrange our data in the buckets we can observe the rows as follows.

# In[174]:


data_as_per_neighborhood['Crime_group'] = pd.cut(data_as_per_neighborhood['Percentage of Crimes'], 3,labels=["Low Crime", "Medium Crime", "High Crime"]) #Adding a category label to the data and splitting into bins  


# In[175]:


data_as_per_neighborhood.sort_values(by='Percentage of Crimes',ascending=False).head(10) # Checking to see how the mapping of neighborhood works


# With the crime percentage and binning at hand we move forward to add it to our dataframe. After this we extract latitude, longitude and Crime group to be inserted into X an Y dataframes respectively to bifurcate the data and create the test and train datasets for the classifier we need train.
# 
# Now we get to a point where we need to choose our classifier. Since we already have the train and test data and need to estimate the coordinate's(latitude, longitude) vicinity to the nearest bin, we need an estimator that takes in another estimator(in this case latitude and longitude) as the parameter. This leads us to using a meta-estimator and since we have multiple models, we need to use the ensemble modelling technique. Moving ahead with this choice we are also conscious to keep the data accurate, reduce variance and keep the inference of noise with our prediction to a minimum. Since our clssifications is not binary, we use Boosting, an ensemble modelling technique to convert our weak learners to strong learners. A specific type of boosting is adaptive boosting, which renders equal weightage to all points until they are not classified correctly, if so then higher weightage is assigned to the wrongly classified data point. Now in the next iteration of the model the points with high weights are prioritised and reallocated their weights the model keeps repeating the logic until it receives a low error value. This sequential process of measuring the error estimates allows us to obtain classification with the minimum number of possible iterations.
# 
# 

# In[176]:


crime_group_dict = dict(zip(data_as_per_neighborhood.index,data_as_per_neighborhood.Crime_group)) #Creating a dicionary that can be used for mapping in the main dataframe


# In[177]:


data_as_per_neighborhood.sort_values(by='Percentage of Crimes',ascending=False).head(10) #Checking if the mapping worked properly


# In[180]:


data_as_per_neighborhood['Crime_group'].value_counts()# Understanding the distribution of the data set


# Now we will use the dictionary created to make mapping of the crime category to the main data set

# In[181]:


data_for_prediction['Crime_group'] = data_for_prediction['NEIGHBORHOOD_ID'].map(crime_group_dict) 


# Like mentioned earlier in the dataset description Null values will exist in the Lattitide and Longitude becasue these are for the sensitive crimes like Sexual Assault

# In[182]:


data_for_prediction.dropna(axis=0,how='any',subset=['GEO_LAT','GEO_LON','Crime_group'],inplace=True) #Dropping Null values


# In[183]:


#Splitting the data into the independent and dependent variables
X = data_for_prediction[['GEO_LAT','GEO_LON']]  #setting the independent variable
y = data_for_prediction['Crime_group'] #setting the dependent variable


# In[184]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #Splitting the data into training and testing into 70 training and 30 testing


# In[326]:


ada_boost_classifier = AdaBoostClassifier(n_estimators=500,learning_rate=1)
# Train Adaboost Classifer
model = ada_boost_classifier.fit(X_train.values, y_train.values)

#Predict the response for test dataset
y_pred = model.predict(X_test.values)


# In[185]:


#Predict the response for test dataset
y_pred = model.predict(X_test.values)


# In[190]:


print(f"Accuracy of the Model: {metrics.accuracy_score(y_test, y_pred)*100:.2f}%") #Checking Accuracy of the Model 


# In order to avoid training the model again and again we save the model as a pickel file.

# In[191]:


filename = 'classifier_model.sav' #File name to save the trained model  


# In[329]:


# save the model to local machine using pickel 
pickle.dump(model, open(filename, 'wb'))


# In[92]:


model = pickle.load(open(filename, 'rb')) 


# Building the Validation data set 

# In[261]:


accident_data['Crime_group'] = accident_data['neighborhood_id'].map(crime_group_dict) #Mapping each neighbhood as High,Low or Medium crime in our data frame


# In[262]:


accident_data.dropna(axis=0,how='any',subset=['geo_lat','geo_lon','Crime_group'],inplace=True) #Ensuring there are no null values


# In[263]:


validation_data_set = accident_data[['geo_lat','geo_lon','Crime_group']].copy() #Making the real data set


# In[264]:


#Splitting the data into the independent and dependent variables
X_valid = validation_data_set[['geo_lat','geo_lon']]  #setting the independent variable
y_valid = validation_data_set['Crime_group'] #setting the dependent variable


# In[265]:


X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split(X_valid, y_valid, test_size=0.3) #Splitting the data into training and testing into 70 training and 30 testing


# In[267]:


y_pred2 = model.predict(X_test_valid.values) #


# In[268]:


print(f"Accuracy of the Model: {metrics.accuracy_score(y_test_valid, y_pred2)*100:.2f}%")


# In[85]:


output_df = pd.DataFrame(data=np.column_stack((y_test,y_pred)),columns=['Expected','Predicted']) # Creating a data frame to see our output


# In[221]:


output_df.sample(n=10)


# ## Observation
# We observe with 73.24 % accuracy that we are successfully able to classsify the latitudes and longitudes into the bins Low Crime High Crime and Medium Crime. This arms us with a powerful information as to how we can identify an area and predict a multitude to coordinates in Denver precisely. Our Classifier works as per the expectation and we can decisively build our inference.

# ## Inference
# 
# It is an unvarnished truth of the modern society, that incidents of crime irrespective of time and place reflects the fragmented social structure of the community. Humans are flawed by nature and thereby every projection of a human life, societies communities and cities included are flawed by definition. At Denver Mifflin, we accept that not every crime occured is an intentional one and are aibiding by the criminal laws laid down federal and state governments, the inferences we offer are solely from an analytical stand point and wholly unbiased. 
# 
# We have two neighborhoods that fall in the high crime region Five Points and Central Park. An individual's coordinates falling under his classifer may face the risk of being a victime of violent crimes. Most neighborhoods come under Low crime classifier which is already a step towards the right direct with only 10 Medium Crime neighborhoods. These neighborhoods have a higher density of crimes compared to their surrounding areas and the more reliable data we feed to the model the more accurate will be the range of ourpredictions.

# ## Recommendations
# 
# We recommend the Departmnment of Police in Denver to caliberate the model with a more comprehensive dataset, this will yield model predictions with even greater precision. We did mention early on that we wanted to arrive at win-win scenario and here is the ingenious part. The same model can be used by the police personnel stationed at any junction across Denver and and based on the priority(High,Low and Medium) an officer can prepare themselves and reducethe response time and strategise which distress call to pick up first.
# 
# With repeated use of this model, more law enforcement officers can be tactically positioned in higher crime density areas which may lead to a lower crime activity. Should any planned criminal activity occur who might shift to other locations due to the intervention of the police while using this model. The more recent data that the model classifies, it will be able to show a shift of planned criminal activities as well. The Department of police in Denver may find it empowering and can also keep a track record of shifting criminal bases.

# # Conclusion

# Our group of consultants uncovered startling evidence that crimes are being committed in Denver on a daily basis. At Denver Mifflin, we were able to connect our research findings to analysis from magazines, newspapers, and mass media organizations. <br>
# Our team of consultants was able to offer the Denver Police Department useful information that they can use to develop protocols to prevent crime in the future by identifying patterns in crime occurrences and researching the characteristics of crime with regard to time, place, and type of crime. Our team at Denver Mifflin has gathered data from a variety of sources in addition to the results of our analysis in order to gain domain knowledge and base recommendations on this literature. To empower the people of Denver, we are proud to present our classifier model that predicts crime density by neighborhood. The prediction model will yield data with increased precision when more data is fed to model in the upcoming years. We thank the Denver Police Department for their confidence in us and for entrusting us with helping them fight crime in the city as we draw to a close.

# # Refrences

# https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-crime <br>
# https://www.westword.com/news/denver-is-one-big-car-theft-hot-spot-15306202 <br>
# https://kdvr.com/news/data/denver-homeless-population-point-in-time-count-2022/ <br>
# https://www.9news.com/article/news/crime/reasons-behind-colorado-car-thefts/73-603b4e70-6329-4600-8238-769c0e73d5c8 <br>
# https://www.adtsecurity.com.au/blog/the-mind-of-a-burglar/ <br>
# https://www.brennancenter.org/our-work/analysis-opinion/sexual-assault-remains-dramatically-underreported  <br>
