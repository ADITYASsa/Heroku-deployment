#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pickle_in = open("microsoft.pickle","rb")
data = pickle.load(pickle_in)


# In[3]:


data.head()


# In[4]:


data.info()


# # MODELLING

# In[5]:


x = data.loc[: , data.columns!= 'Close']
x


# In[6]:


y = data['Close']
y


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[10]:


x_train


# # SVM LINEAR

# In[11]:


from sklearn.svm import SVR
msftlinear = SVR(kernel = 'linear' , C = 10)


# In[12]:


msftlinear.fit(x_train , y_train)


# In[13]:


y_train_pred = msftlinear.predict(x_train)


# In[14]:


y_test_pred = msftlinear.predict(x_test)


# In[15]:


result = pd.DataFrame({'Actual' : y_test , 'Predicted' : y_test_pred}).reset_index()[['Actual' , 'Predicted']]
result


# In[16]:


plt.plot(y_test,result['Predicted'],color='red')
plt.plot(y_test,result['Actual'],color='blue')


# In[17]:


from sklearn.metrics import mean_squared_error, r2_score


# In[18]:


mse = mean_squared_error(y_test,y_test_pred)
print("Mean_squared_error is: ", mse)


# In[19]:


r2score = r2_score(y_test , y_test_pred)
print("R2 score is: " , r2score)


# # SVM POLYNOMIAL

# In[20]:


from sklearn.svm import SVR
msftpoly = SVR(kernel = 'poly' , C = 10)


# In[21]:


msftpoly.fit(x_train , y_train)


# In[22]:


y_train_pred = msftpoly.predict(x_train)


# In[23]:


y_test_pred = msftpoly.predict(x_test)


# In[24]:


result = pd.DataFrame({'Actual' : y_test , 'Predicted' : y_test_pred}).reset_index()[['Actual' , 'Predicted']]


# In[25]:


plt.plot(y_test,result['Predicted'],color='pink')
plt.plot(y_test,result['Actual'],color='blue')


# In[26]:


from sklearn.metrics import mean_squared_error, r2_score


# In[27]:


mse = mean_squared_error(y_test,y_test_pred)
print("Mean_squared_error is: ", mse)


# In[28]:


r2score = r2_score(y_test , y_test_pred)
print("R2 score is: " , r2score)


# In[30]:


pickle_out = open("microsoft_model.pkl","wb")
pickle.dump(data, pickle_out)
pickle_out.close()


# In[ ]:




