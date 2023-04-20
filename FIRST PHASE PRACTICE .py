#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Prediction Project

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("winequality-red.csv")
df


# There are 1599 rowa and 12 columns in the data set.These data set contains only numerical value with "quality" column as target variable

# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# There are two data types present in the data set i.e float and int

# In[6]:


df.isnull().sum()


# There is no null values present in the data set

# In[7]:


df.info()


# There are two data types present inn the data set 
# i.e float and int64

# In[8]:


sns.heatmap(df.isnull())


# there is no null values present in the dataset

# In[9]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')
    


# There are the value count of all the columns

# In[10]:


df.nunique().to_frame("No. of unique values")


# In[11]:


df["quality"].value_counts()


# quality 7 or higher than 7 considers in qood quality wine

# In[12]:


# checing duplicate value
print("Total duplicate value present in the data set : ",df.duplicated().sum())


# There are 240 duplicate values present in the data set

# In[13]:


# removing dulplicate value
Df =df.drop_duplicates(subset=None , keep='first' , inplace=False, ignore_index=False)


# In[14]:


Df.describe()


# In[15]:


# there is no missing data present in the data set .We can obervse there is a huge difference between 3rd Quartilr(75%) and max value means outliers are present in the data set.


# In[16]:


plt.figure(figsize=(22,10))
sns.heatmap(df.describe(), annot=True,fmt='0.2f',linewidth=0.2,linecolor='black',cmap='Spectral')
plt.xlabel('Figure',fontsize=14)
plt.ylabel('Feature_names',fontsize=14)
plt.title('Descriptive Graph',fontsize=20)
plt.show()


# In[17]:


Df


# In[18]:


sns.pairplot(data=Df,palette="m")


# In[19]:


# Univariate analysis
sns.countplot(x='fixed acidity',data=Df)
print(Df["fixed acidity"].value_counts())


# In[20]:


numerical_col=[]
for i in Df.dtypes.index:
    if Df.dtypes[i]!="object":
        numerical_col.append(i)
print("Numerical Columns : ", numerical_col)


# In[21]:


plt.figure(figsize=(10,6),facecolor="white")
plotnumber = 1
for col in numerical_col:
    if plotnumber<=12:
        ax=plt.subplot(3,4,plotnumber)
        sns.boxplot(df[col],color="m")
        plt.xlabel(col,fontsize=12)
        plt.yticks(rotation=0,fontsize=10)
    plotnumber+=1
plt.tight_layout()


# In[22]:


from scipy.stats import zscore


# In[23]:


outliers= Df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
z=np.abs(zscore(outliers))
z


# In[24]:


np.where(z>3)


# In[25]:


df1=Df[(z<3).all(axis=1)]


# In[26]:


df1


# In[27]:


print("Old Data : ",df.shape[0])
print(" Data after duplicate removed : ",Df.shape[0])
print("New Data : ",df1.shape[0])


# In[28]:


print("Data Loss percentage : ",((df.shape[0]-df1.shape[0])/df.shape[0]))


# In[29]:


df1.skew()


# There is high skewness present in residual sugar colomn , let's remove skewness by using cuberoot method

# In[30]:


df1["residual sugar"] = np.cbrt(df1["residual sugar"])


# In[31]:


df1.skew()


# In[32]:


df1["residual sugar"] = np.cbrt(df1["residual sugar"])


# In[33]:


df1.skew()


# In[34]:


df1['chlorides']=np.cbrt(df1['chlorides'])


# In[35]:


df1.skew()


# In[36]:


sns.distplot(df["chlorides"], color='m' , kde_kws = {"shade" : True},hist=False)


# In[37]:


df1['total sulfur dioxide']=np.cbrt(df1['total sulfur dioxide'])


# In[38]:


df1.skew()


# In[39]:


df1.corr()


# In[40]:


df1.corr().quality.sort_values()


# In[41]:


plt.figure(figsize=(22,7))
df.corr()['quality'].sort_values(ascending= False).drop(['quality']).plot(kind='bar',color='m')
plt.xlabel('Feature',fontsize = 15)
plt.ylabel('Target',fontsize = 15)
plt.title("Correlation between Feature and Target using bar plot", fontsize=20)


# In[43]:


for i in df1['quality']:
    if i>=7:
        print(" wine quality is Good")
    else:
        print("Wine quality is not good")


# In[ ]:




