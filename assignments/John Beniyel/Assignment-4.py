#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


df = pd.read_csv(r"E:\csv\Mall_Customers.csv")


# In[16]:


df.head()


# # Univariable Analysis

# In[17]:


df["Annual Income (k$)"].plot(kind='hist');


# # Bi-Variate Analysis

# In[18]:


cy=df[df.Gender	=="Male"].Age
cn=df[df.Gender=="Female"].Age
plt.xlabel("Age")
plt.ylabel("No of peoples")
plt.hist([cy,cn],color=['blue','red'],label=["Height=yes"])
plt.show()


# # Multi-variate Analysis

# In[19]:


sns.pairplot(df)


# # Descriptive statistics

# In[20]:


df.describe()


# # Handle The Missing values

# In[21]:


df.isnull().any()


# In[22]:


df.isnull().sum()


# # Find the outliers

# In[23]:


df.skew()


# # Split data into dependent and independent variables

# In[24]:


x=df.iloc[:,3:13].values
y=df.iloc[:,13:14].values
x.shape


# In[25]:


y.shape


# # Categorical colums and encoding

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([("oh",OneHotEncoder(),[1,2])],remainder="passthrough")
x=ct.fit_transform(x)
x.shape


# In[27]:


df["Gender"].unique()


# # Split the data into training and testing

# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape


# In[29]:


x_test.shape


# # Scale the independent variables

# In[30]:


X = df.iloc[:, :-1].values
print(X)


# In[31]:


import joblib
joblib.dump(ct,"Mall_Customers.pkl")


# In[ ]:


#Split the data into training and testing 
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
#print(train)
#print(test)


# In[33]:


# Build the Model 
my_dict=pd.read_csv(r"E:\csv\Mall_Customers.csv")
df = pd.DataFrame(my_dict)
print(df)


# In[36]:


#  Build the Model 
import csv
with open(r"E:\csv\Mall_Customers.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    df = pd.DataFrame([csv_reader], index = None)
for val in list(df[1]):
    print(val)


# In[ ]:


# Training and Testing Module 
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
print(train)
print(test)


# In[ ]:


#Measure the performance using matrics
from __future__ import print_function

import pandas as pd
path = "/abalone.csv"
merged = pd.read_csv(path, error_bad_lines=False, low_memory=False)

X = merged.text
y = merged.grid

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred_class))

