#!/usr/bin/env python
# coding: utf-8

# # Import Libararies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 1: Import dataset¶

# In[2]:


file_path = r"C:\Users\kesho\Downloads\P74-Project-1\Car_Purchasing_Data.csv"
car_df = pd.read_csv(file_path, encoding = 'ISO-8859-1')


# In[3]:


car_df.head(5)


# # step 2: Visualize data

# In[4]:


sns.pairplot(car_df)


# # step 3: Create Testing and Training dataset/Data cleaning

# In[5]:


X = car_df.drop(['Customer Name','Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[6]:


X


# In[7]:


y = car_df['Car Purchase Amount']


# In[8]:


y


# In[9]:


X.shape


# In[10]:


y.shape


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[12]:


scaler.data_max_


# In[13]:


scaler.data_min_


# In[14]:


y = y.values.reshape(-1,1)


# In[15]:


y_scaled = scaler.fit_transform(y)


# In[16]:


y_scaled


# # Step 4: Training the Model

# In[17]:


X_scaled.shape


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25 )


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))


# In[22]:


model.summary()


# In[23]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[24]:


epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, verbose = 1, validation_split = 0.2)


# # Step 5 Evaluation

# In[25]:


epochs_hist.history.keys()


# In[26]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss progress during training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[35]:


# Gender, Age, Annuaul Salary, Credit Card Debt, Net Worth
X_test = np.array([[1, 50, 50000, 10000, 600000]])
y_predict = model.predict(X_test)


# In[36]:


print('Expected Purchase Amount', y_predict)


# In[ ]:




