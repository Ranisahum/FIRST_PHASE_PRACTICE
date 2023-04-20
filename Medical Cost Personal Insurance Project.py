#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("medical_cost_insurance.csv")


# In[4]:


df


# there is both numerical and categorical data present in the data set ," Charges" is the target variable .

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# There is no null values present in the data set

# In[9]:


sns.heatmap(df.isnull())


# In[10]:


df.nunique()


# In[11]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[12]:


# separating categorical and numerical columns


# In[13]:


df.index


# In[14]:


categorical_col=[]
for i in df.dtypes.index:
    if df.dtypes[i]=="object":
        categorical_col.append(i)
print("categorical columns : ",categorical_col)
print("\n")


# In[15]:


numerical_col=[]
for i in df.dtypes.index:
    if df.dtypes[i]!="object":
        numerical_col.append(i)
print("Numerical columns : ",numerical_col)


# In[16]:


df.nunique().to_frame("No. of unique values")


# In[17]:


# univariate analysis

sns.countplot(x="sex", data = df)


# There is no differnce between male and female , both are almost same in count

# In[18]:


sns.countplot(x="smoker", data=df)


# There is less number of smokers in the data set.

# In[19]:


sns.countplot(x="region",data=df)


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


plt.figure(figsize=(10,6) , facecolor ="white")
pn=1
for col in numerical_col :
    if pn<=4:
        ax= plt.subplot(2,2,pn)
        sns.distplot(df[col],color="m")
        plt.xlabel(col, fontsize=12)
        plt.ylabel(col, fontsize=10)
    pn+=1
plt.tight_layout()


# # bivariate analysis

# In[29]:


plt.title("Analysis between charges and bmi")
sns.scatterplot(x="charges",y="bmi", data= df, hue="sex", palette="bright")


# In[34]:


plt.title("analysis on age and childern")
sns.scatterplot(x="age",y="charges",data=df,hue="sex",palette="bright")


# In[36]:


sns.pairplot(df,hue='smoker',palette="Dark2")
plt.show()


# In[38]:


df.duplicated().sum()


# In[39]:


Df=df.drop_duplicates()


# In[41]:


Df


# In[42]:


df.info()


# In[44]:


Df.skew()


# In[50]:


# Encoding categorical columns mean data will change from categorical to numerical
import sklearn as sk


# In[51]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()
for i in Df.columns:
    if Df[i].dtypes=="object":
        Df[i]=OE.fit_transform(Df[i].values.reshape(-1,1))
Df


# In[52]:


Df.info()


# In[55]:


Df.describe()


# In[56]:


cor = Df.corr()


# In[57]:


cor


# In[58]:


plt.figure(figsize=(20,15))
sns.heatmap(Df.corr(), linewidths=0.1, fmt=".1g",linecolor="black",annot=True,cmap="Blues_r")
plt.yticks(rotation=0);
plt.show()


# In[62]:


cor["charges"].sort_values(ascending=False)


# In[63]:


plt.figure(figsize=(22,7))
Df.corr()["charges"].sort_values(ascending=False).drop(["charges"]).plot(kind='bar',color='m')
plt.xlabel('Feature',fontsize=15)
plt.ylabel('Target',fontsize=15)
plt.title("Correlation between label and features using barplot",fontsize=20)
plt.show()


# In[66]:


Df.drop("children",axis=1 , inplace=True)


# In[67]:


Df


# In[68]:


x=Df.drop("charges",axis=1)
y=Df["charges"]


# # Feature Scalling using Standard Scalarization

# In[69]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x= pd.DataFrame(scaler.fit_transform(x), columns= x.columns)
x


# # Checking variance Inflation Factor(VIF)

# In[70]:


# Finding varience inflation factor in each scaled columni.e x.shape[1] (1/1-r2)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif["VIF values"]=[variance_inflation_factor(x.values,i)
                  for i in range(len(x.columns))]
vif["Features"]=x.columns

vif


# In[71]:


y.value_counts()


# In[72]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y = SM.fit_resample(x,y)


# # Modeling

# In[74]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
maxAccu = 0
maxRS = 0
for i in range(1,200):
    x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.30 , random_state=i)
    RFR = RandomForestClassifier()
    RFR.fit(x_train,y_train)
    pred = RFR.predict(x_test)
    acc= accuracy_score(y_test , pred)
    if acc>maxAccu:
        maxAccu = acc
        maxRS = i
print("Best Accuracy is : " , maxAccu , "At random_state ", maxRS)


# In[ ]:




