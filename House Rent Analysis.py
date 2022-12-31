#!/usr/bin/env python
# coding: utf-8

# In[113]:





# In[114]:


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype as is_num
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# In[115]:


#Importing the CSV file using Pandas

house_rent_df = pd.read_csv("House_Rent_Dataset.csv", delimiter = ',', header = 'infer')
house_rent_df


# In[116]:


house_rent_df.describe()


# In[117]:


# Total no of null values for each col

house_rent_df.isnull().sum()


# In[118]:


# A function to show statistical info about every column

def show_stats(df):
    
    stats = pd.DataFrame(columns = ['Count', 'Dtype', 'Unique', 'Numeric', 'min', '25%', 'median', 'mode', 'mean', '75%', 'max', 'skew', 'kurt', 'std'])
    
    for col in df:
        
        count = df[col].count()                 # no. of entries for each feature
        dtype = df[col].dtype                   # data type 
        nunique = df[col].nunique()             # no. of unique values
        num = is_num(df[col])
        
        
        # For numeric cols calculate min, max, mean, median,...
        if is_num(df[col]):
            
            # Calculating mathematical statistics for numeric columns of datset
            min_val = df[col].min()
            max_val = df[col].max()
            first_q = df[col].quantile(0.25)   # first quartile
            median = df[col].median()
            mode = df[col].mode().values[0]
            mean = df[col].mean()
            third_q = df[col].quantile(0.75)   # third quartile
            skewness = df[col].skew()
            kurt = kurtosis(df[col])
            std = df[col].std()
            
            #inserting rows into the output dataframe
            stats.loc[col] = [count, dtype, nunique,num, min_val, first_q , median, mode, 
                              mean, third_q, max_val, skewness, kurt, std]
        
        # For non-numeric cols, show total count, no. of unique values
        else:
            stats.loc[col] = [count, dtype, nunique,num, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    
    return stats.sort_values(by = ['Numeric', 'skew', 'Unique'], ascending = False)


# In[119]:


# Statistical information for cols in the house rent dataset

show_stats(house_rent_df)


# In[120]:


house_rent_df['Floor'].to_string()


# In[121]:



# Converting the column "Floor" from string dtype into 2 new numeric columns: "Floor Level" and "Max Floor"
new_df = house_rent_df
# Creating two arrays to store Floor level and the top floor of the building
floor_level = []
max_floor = []
for (key, value) in new_df['Floor'].iteritems():
    floor = value.split()
    
    if floor[0] == 'Ground':           # If floor level is ground, assign 1
        floor_level.append(1)
           
    elif floor[0] == 'Upper':          # If floor level is Upper Basement, assign 0.75 
        floor_level.append(0.75)
           
    elif floor[0] == 'Lower':          # If floor level is Lower Basement, assign 0.25 
        floor_level.append(0.25)
         
    else:
        floor_level.append(int(floor[0])+1)       # Otherwise convert the floor level from str to int  
            
    if floor[-1] == 'Ground':
        max_floor.append(1)
        
    else:
        max_floor.append(int(floor[-1])+1)
        
# Storing new numeric values to col Floor level and Max Floor
new_df['Floor Level'] = floor_level
new_df['Max Floor'] = max_floor

new_df[['Floor', 'Floor Level', 'Max Floor']]


# In[122]:


# Keeping only the columns that we intend to use in our analysis

new_df.drop(['Area Locality'], axis=1, inplace = True)

new_df.drop(['Floor'], axis=1, inplace = True)

new_df.drop(['Posted On'], axis=1, inplace = True)

new_df
# show_stats(new_df)


# In[123]:


# Identifying outliers for Rent

def show_boxplots(df):

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.suptitle("")
    fig.set_figheight(13)
    fig.set_figwidth(14)
        
    ax1.boxplot(df['Rent'])
    ax1.set_title("Rent")

    ax2.boxplot(df['Size'])
    ax2.set_title("Size")

    ax3.boxplot(df['BHK'])
    ax3.set_title("BHK")

    ax4.boxplot(df['Bathroom'])
    ax4.set_title("Bathroom")

    ax5.boxplot(df['Floor Level'])
    ax5.set_title("Floor Level")

    ax6.boxplot(df['Max Floor'])
    ax6.set_title("Max Floor")
    
show_boxplots(new_df)

# Heading-2: Exploratory Data Analysis

# In this section we are finding outliers by plotting subplots of each column present in the original dataset. The small circles
# present at the extreme top of the boxplot can be said as the outliers. 


# In[124]:


df1 = new_df

# Removing outliers from the columns "Rent" and "Size"
for col in ['Rent', 'Size']:
    
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    IQR = q3-q1
    lower_outliers = q1 - (1.5 * IQR) 
    higher_outliers = q3 + (1.5 * IQR) 

    df1 = df1[df1[col] < higher_outliers]
    df1 = df1[df1[col] > lower_outliers]

# Distributions after removing outliers
show_boxplots(df1)
show_stats(df1)


# In[125]:


# Counting the no. of values for each unique value in "Area Type", "Point of Contact",
# and "Tenant Preferred

print(df1['Area Type'].value_counts()), 
print('\n',df1['Point of Contact'].value_counts()), 
print('\n',df1['Tenant Preferred'].value_counts()), 
print('\n',df1['Furnishing Status'].value_counts())


# In[126]:


# Replacing column subtype with extremely low counts with the mode

df1['Area Type'].replace({'Built Area': 'Super Area'}, inplace=True)

df1['Point of Contact'].replace({'Contact Builder': 'Contact Owner'}, inplace=True)

print(df1['Area Type'].value_counts()), 
print('\n',df1['Point of Contact'].value_counts()), 


# In[127]:


# Q1:  Relationship of Rent with Size
# 1-1: Distribution of Rent and Size with a culmulative distribution curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4))

sns.histplot(data=df1, x="Rent", ax=ax1)
sns.histplot(data=df1, x="Size", ax=ax2)


# In[128]:


# 1-2: Regression graph used to show how rent varies with size

sns.lmplot(x="Size", y="Rent", data=df1,
lowess=True, scatter=False,line_kws={"color": "C1"});


# In[129]:


# Q2: Which city is the most expensive to live in? (in terms of house rent affordability)

# 2-1: Distribution of Rent accross different cities

sns.catplot(data=df1, x="City", y="Rent", kind="box")


# In[130]:


# 2-2: How Rent changes with the size of a house in different cities

sns.lmplot(x="Size", y="Rent", data=df1, hue="City",
           lowess=True, scatter=False);


# In[131]:


# Q3: Does the floor level of a house affect its rent?

# 3-1: Distribution of Rent and Floor levels in the dataset

sns.displot(data=df1, x="Rent")
sns.displot(data=df1, x="Floor Level")


# In[132]:


# 3-2: Relationship between Rent and Max Floor

sns.lmplot(x="Max Floor", y="Rent", data=df1,
           lowess=True, scatter=False, line_kws={"color": "C1"});


# In[133]:


# Q4: How do the houses in each city compare with each other in terms of the size of the house and the number of floors?

# 4-1: Distribution of Max Floor levels accross different cities

sns.catplot(data=df1, x="City", y="Max Floor", kind="boxen")


# In[134]:


# 4-2: Distribution of house sizes accross different cities

sns.catplot(data=df1, x="City", y="Size", kind="boxen")


# In[135]:


# Converting categorical columns into numeric data types by assigning numeric labels to string values

cols_to_replace = {'Area Type'        : {'Super Area': 1, 'Carpet Area': 2, 'Built Area': 3}, 
                   
                   'Point of Contact' : {'Contact Owner': 1, 'Contact Agent': 2, 'Contact Builder': 3}, 
                   
                   'Tenant Preferred' : {'Bachelors/Family': 1, 'Bachelors': 2, 'Family': 3},
                   
                   'Furnishing Status': {'Unfurnished': 1, 'Semi-Furnished': 2, 'Furnished': 3},
                   
                   'Area Type'        : {'Super Area': 1, 'Carpet Area': 2, 'Built Area': 3},
                   
                   'City'             : {'Kolkata': 1, 'Mumbai': 2, 'Bangalore': 3, 'Delhi': 4, 'Chennai': 5, 'Hyderabad': 6}}

df2 = df1.replace(cols_to_replace)


# In[136]:


plt.figure(figsize=(12,4))
sns.heatmap(df2.corr(), annot = True)


# In[137]:


# Scaling values using scikit learn standard scaler

ss = StandardScaler()

cols_to_scale = ['BHK', 'Rent', 'Size', 'Area Type'
                 , 'City', 'Furnishing Status', 'Tenant Preferred',
       'Bathroom', 'Point of Contact', 'Floor Level', 'Max Floor']
df3 = df2

df3[cols_to_scale] = ss.fit_transform(df2[cols_to_scale])
df3


# In[138]:


# Splitting features into independent and dependent variables

X = df3[['BHK', 'Size', 'Area Type', 'Bathroom', 'City', 'Point of Contact', 'Floor Level', 'Max Floor']]
Y = df3['Rent']


# In[139]:


# Splitting the cleaned dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=42)

# Loading the gradient boost regression model for training
reg = GradientBoostingRegressor(learning_rate=0.15, random_state=42, max_depth = 5, n_estimators=60)

# Using the model to train on our training data
reg.fit(X_train, y_train)
""
# Generating predictions using the trained model on the test set
ypred = reg.predict(X_test)

# Calculating the accuracy/score of the model
reg.score(X_test,y_test)


# In[ ]:




