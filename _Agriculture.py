#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np               
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from ipywidgets import interact   


# In[4]:


data = pd.read_csv('data.csv')


# In[6]:


print("Shape of the dataset:", data.shape)


# In[7]:


data.head()


# In[8]:


data.isnull().sum()


# In[9]:


data['label'].value_counts()

print("Average Ratio of Nitrogen in the Soil: {:.2f}".format(data["N"].mean()))
print("Average Ratio of Phosphorous in the Soil: {:.2f}".format(data["P"].mean()))
print("Average Ratio of Potassium in the Soil: {:.2f}".format(data["K"].mean()))
print("Average Temperature in Celsius: {:.2f}".format(data["temperature"].mean()))
print("Average Relative Humidity in %: {:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the soil: {:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm: {:.2f}".format(data["rainfall"].mean()))
# In[10]:


print("Average Ratio of Nitrogen in the Soil: {:.2f}".format(data["N"].mean()))
print("Average Ratio of Phosphorous in the Soil: {:.2f}".format(data["P"].mean()))
print("Average Ratio of Potassium in the Soil: {:.2f}".format(data["K"].mean()))
print("Average Temperature in Celsius: {:.2f}".format(data["temperature"].mean()))
print("Average Relative Humidity in %: {:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the soil: {:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm: {:.2f}".format(data["rainfall"].mean()))


# In[11]:


@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    
    print("***")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required:", x['N'].min())
    print("Average Nitrogen required:", x['N'].mean())
    print("Maximum Nitrogen required:", x['N'].max())
    print("***")
    print("***")
    print("Statistics for Phosphorus")
    print("Minimum Phosphorus required:",x['P'].min())
    print("Average Phosphorus required:",x['P'].mean())
    print("Maximum Phosphorus required:",x['P'].max())
    print("***")
    print("***")
    print("Statistics for Potassium")
    print("Minimum Potassium required:",x['K'].min())
    print("Average Potassium required:",x['K'].mean())
    print("Maximum Potassium required:",x['K'].max())
    print("***")
    print("***")
    print("Statistics for Temperature")
    print("Minimum Temperature required: {:.2f}".format(x["temperature"].min()))
    print("Average Temperature required: {:.2f}".format(x["temperature"].mean())) 
    print("Maximum Temperature required: {:.2f}".format(x["temperature"].max()))
    print("***")
    print("***")
    print("Statistics for Humidity")
    print("Minimum Humidity required: {:.2f}".format(x['humidity'].min()))
    print("Average Humidity required: {:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required: {:.2f}".format(x['humidity'].max()))
    print("***")
    print("***")
    print("Statistics for PH")
    print("Minimum PH required: {:.2f}".format(x['ph'].min()))
    print("Average PH required: {:.2f}".format(x['ph'].mean()))
    print("Maximum PH required: {:.2f}".format(x['ph'].max()))
    print("***")
    print("***")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required: {:.2f}".format(x["rainfall"].min()))
    print("Average Rainfall required: {:.2f}".format(x["rainfall"].mean()))
    print("Maximum Rainfall required: {:.2f}".format(x["rainfall"].max()))




# In[12]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average value for",conditions,"is{:.2f}".format(data[conditions].mean()))
    print("***")
    print("Rice :{:.2f}".format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Maize :{:.2f}".format(data[(data['label'] == 'maize')][conditions].mean()))
    print("Jute :{:.2f}".format(data[(data['label'] == 'jute')][conditions].mean()))
    print("Cotton :{:.2f}".format(data[(data['label'] == 'cotton')][conditions].mean()))
    print("Coconut :{:.2f}".format(data[(data['label'] == 'coconut')][conditions].mean()))
    print("Papaya :{:.2f}".format(data[(data['label'] == 'papaya')][conditions].mean()))
    print("Orange :{:.2f}".format(data[(data['label'] == 'orange')][conditions].mean()))
    print("Apple :{:.2f}".format(data[(data['label'] == 'apple')][conditions].mean()))
    print("Muskmelon :{:.2f}".format(data[(data['label'] == 'muskmelon')][conditions].mean()))
    print("Watermelon :{:.2f}".format(data[(data['label'] == 'watermelon')][conditions].mean()))
    print("Grapes :{:.2f}".format(data[(data['label'] == 'grapes')][conditions].mean()))
    print("Mango :{:.2f}".format(data[(data['label'] == 'mango')][conditions].mean()))
    print("Banana :{:.2f}".format(data[(data['label'] == 'banana')][conditions].mean()))
    print("Pomegrante :{:.2f}".format(data[(data['label'] == 'pomegrante')][conditions].mean()))
    print("Lentil :{:.2f}".format(data[(data['label'] == 'lentil')][conditions].mean()))
    print("Blackgram :{:.2f}".format(data[(data['label'] == 'blackgram')][conditions].mean()))
    print("Mungbean :{:.2f}".format(data[(data['label'] == 'mungbean')][conditions].mean()))
    print("Mothbeans :{:.2f}".format(data[(data['label'] == 'mothbeans')][conditions].mean()))
    print("Pigeonpeas :{:.2f}".format(data[(data['label'] == 'pigeonpeas')][conditions].mean()))
    print("Kidneybeans :{:.2f}".format(data[(data['label'] == 'kidneybeans')][conditions].mean()))
    print("Chickpea :{:.2f}".format(data[(data['label'] == 'chickpea')][conditions].mean()))
    print("Coffee:{:.2f}".format(data[(data['label'] == 'coffee')][conditions].mean()))
    


# In[13]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", conditions, '\n')
    print(data[data[conditions]> data[conditions].mean()]['label'].unique())
    print("Crops which require less than averag", conditions, '\n')
    print(data[data[conditions]<= data[conditions].mean()]['label'].unique())
  

   

 

 

 



# In[14]:


plt.subplot(3,4,1)
sns.histplot(data['N'], color="yellow")
plt.xlabel('Nitrogen', fontsize = 12)
plt.grid()

plt.subplot(3,4,2)
sns.histplot(data['P'], color="orange")
plt.xlabel('Phosphorous', fontsize = 12)
plt.grid()

plt.subplot(3,4,3)
sns.histplot(data['K'], color="darkblue")
plt.xlabel('Pottasium', fontsize = 12)
plt.grid()

plt.subplot(3,4,4)
sns.histplot(data['temperature'], color="black")
plt.xlabel('Temperature', fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.histplot(data['rainfall'], color="grey")
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.histplot(data['humidity'], color="lightgreen")
plt.xlabel('Humidity', fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.histplot(data['ph'], color="darkgreen")
plt.xlabel('PH Level', fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
plt.show()


# In[37]:


print("Some Interesting Patterns")
print("...........................................")
print("Crops that require very High Ratio of Nitrogen Content in Soil:", data[data['N'] > 120]['label'].unique())
print("Crops that require very High Ratio of Phosphorous Content in Soil:", data[data['P'] > 100]['label'].unique())
print("Crops that require very High Ratio of Potassium Content in Soil:", data[data['K'] > 200]['label'].unique())
print("Crops that require very High Rainfall:", data[data['rainfall'] > 200]['label'].unique())
print("Crops that require very Low Temperature:", data[data['temperature'] < 10]['label'].unique())
print("Crops that require very High Temperature:", data[data['temperature'] > 40]['label'].unique())
print("Crops that require very Low Humidity:", data[data['humidity'] < 20]['label'].unique())
print("Crops that require very Low pH:", data[data['ph'] < 4]['label'].unique())
print("Crops that require very High pH:", data[data['ph'] > 9]['label'].unique())


# In[15]:


print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("...........................................")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("...........................................")
print("Monsoon Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())


# In[16]:


from sklearn.cluster import KMeans

#removing the labels column
x = data.drop(['label'], axis=1)

#selecting all the values of data
x = x.values

#checking the shape
print(x.shape)


# In[43]:


plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#Plotting the results

plt.plot(range(1,11), wcss)
plt.title('Elbow Method', fontsize = 20)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show


# In[17]:


km = KMeans(n_clusters = 4, init = 'k-means++',  max_iter = 2000, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Finding the results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Checking the clusters for each crop
print("Lets Check the results after applying K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("...........................................")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("...........................................")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("...........................................")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[19]:


y = data['label']
x = data.drop(['label'], axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[20]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The Shape Of x train:", x_train.shape)
print("The Shape Of x test:", x_test.shape)
print("The Shape Of y train:", y_train.shape)
print("The Shape Of y test:", y_test.shape)


# In[18]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[48]:


from sklearn.metrics import confusion_matrix

#Printing the Confusing Matrix
plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'Wistia')
plt.title('Confusion Matrix For Logistic Regression', fontsize = 15)
plt.show()


# In[49]:


#Defining the classification Report
from sklearn.metrics import classification_report

#Printing the Classification Report
cr = classification_report(y_test, y_pred)
print(cr)


# In[51]:


data.head()


# In[50]:


prediction = model.predict((np.array([[90, 40, 40, 20, 80, 7, 200]])))
print("The Suggested Crop for given climatic condition is :",prediction)


# In[ ]:





# In[ ]:




