#!/usr/bin/env python
# coding: utf-8

# # PART ll: Machine Learning: Supervised - Linear Regression

# In[1]:


#Import Python Libraries (NumPy and Pandas)
import pandas as pd
import numpy as np


# In[2]:


#Import modules and libraries for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[3]:


#Import scikit-learn module for the algorithm/model (linear regression)
from sklearn.linear_model import LinearRegression


# In[4]:


#Import scikit-learn module to split the dataset in to train it and test subdatasets
from sklearn.model_selection import train_test_split


# In[5]:


#Import scikit-learn module for K-fold cross-validation (algorithm/model evaluation and validation)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[6]:


#Import scikit-learn module classification report to use for information about how the system try to classify
from sklearn.metrics import classification_report


# # Step 1: Load the data

# In[ ]:





# In[7]:


filename = "/Users/miriamgarcia/Downloads/housing_boston_w_hdrs.csv"
df=pd.read_csv(filename)
print(df)


# # Step 2: Preprocess the dataset

# In[8]:


#Clean data and find any missing values
#From looking at the data above we knew that ZN and CHAS had zeros. 
#Since most are missing values, it is best to drop them entirely. 
df = df.drop("ZN",1)
df = df.drop("CHAS",1)


# In[9]:


#Count the number of NaN values in each
print(df.isnull().sum())


# In[10]:


#Now there is no invalid zero value in any column of the original data.


# # Step 3: Perform the Exploratory Data Analysis (EDA)

# In[11]:


#Get the dimensions/shape of the dataset
# which will give us the number of records/rows x number of variables/columns
print(df.shape)


# In[12]:


# Now find the data types of all variables/attributes of the data set
print(df.dtypes)


# In[13]:


#Get several records/rows at the top of the dataset, we get 5 to get a feel of the data. 
print(df.head(5))


# In[14]:


#We can get the summary statistics of the numeric variables/attributes of the dataset.
print(df.describe())


# In[15]:


#Plot histrogram for each numeric
df.hist(figsize=(12, 8))
pyplot.show()


# In[16]:


#Calculate density plots
# 5 numeric variables -> at Least 5 plots -> Layout (2, 3): 2 rows, each row with 3 plots
df.plot(kind='density', subplots=True, layout=(12, 2), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[17]:


#Calculate box plots
df.plot(kind='box', subplots=True, layout=(12,2), sharex=False, figsize=(12,8))
pyplot.show()


# In[18]:


#Calculate scatter plot matrices
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# # Step 4: Separate the dataset into the input and output NumPy arrays

# In[19]:


#Then we separate the dataset into input and output NumPy arrays
#We want to store the dataframe values into a NumPy array
array = df.values
#Then we want to separate the array into input and output components by slicing it
#For X (input)[:, 5] --> all the rows, columns from 0 - 4 (5 - 1)
X = array[:,0:11]
#And for Y (output)[:, 5] --> all the rows, column index 5 (Last column)
Y = array[:,1]


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


#Now we want to split the dataset --> training sub-dataset: 67%; and test sub-dataset:
test_size = 0.33
# Selection of records to include in which sub-dataset must be done randomly
#and use this seed for randomization
seed = 10
# Split the dataset (both input & outout) into training/testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# # Step 5: Split the input/output arrays into the training/testing datasets

# In[23]:


print(X_train)


# In[24]:


print(Y_train)


# # Step 6: Build and train the model

# In[25]:


#Now we can build the model
model = LinearRegression()
#Then train the model using the training sub-dataset
model.fit(X_train, Y_train)
#Print out the coefficients and the intercept
#Print intercept and coefficients
print (model.intercept_)
print (model.coef_)


# In[26]:


#We can pair the feature names with the coefficients 
#and print out the list with their correspondent variable name
names_2 = ['CRIM','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


# In[27]:


coeffs_zip = zip(names_2, model.coef_)


# In[28]:


#Convert iterator into set
coeffs = set(coeffs_zip)


# In[29]:


#Print (coeffs)
for coef in coeffs:
    print (coef)


# In[30]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# # Step 7: Calculate the R2 value

# In[31]:


R_squared = model.score(X_test, Y_test)
print(R_squared)


# In[32]:


#Here we get a perfect R-Squared score of 1. A higher R-Squared 
#indicates a higher correlatio between the independent variables and the dependent variables. 


# # Step 8: Predict the "Median value of owner-occupied homes in 1000 dollars"

# # Scenario 1: It is assumed that two new suburbs/towns/developments have been established in the Boston area. The agency has collected the housing data of these two new suburbs/towns/developments.

# In[33]:


#We will use the mean values of the described statistics for the first "made up housing records"
#The suburb area has the following predictors:
#CRIM:1.42
#INDUS:10.30
#NOX:0.54
#RM: 6.34
#AGE:65.56 (proportion of owner-occupied units built prior to 1940)
#DIS: 4.04 (weighted distances to five Boston employment centers)
#RAD: 7.82 (index of accessibility to radial highways)
#TAX: 377.44
#PTRATIO: 18.25 (pupil-teacher ratio by town)
#B:369.83
#LSTAT:23.75


# In[34]:


model.predict([[1.42, 10.30, 0.54, 6.34, 65.56,4.04,7.82,377.44,18.25,369.83,23.75]])


# In[35]:


#The model predicts that the median value of owner-occupied homes 
#in 1000 dollars in the above suburb should be around 10,300 under this scenario


# In[36]:


#We will now have a "made up housing record" in order to retrain our model.
#The suburb area has the following predictors:
#CRIM:2.8
#INDUS:11.30
#NOX:0.55
#RM: 6.39
#AGE:45.56 (proportion of owner-occupied units built prior to 1940)
#DIS: 8.04 (weighted distances to five Boston employment centers)
#RAD: 4.82 (index of accessibility to radial highways)
#TAX: 277.44
#PTRATIO: 13.25 (pupil-teacher ratio by town)
#B:269.83
#LSTAT:21.75


# In[37]:


model.predict([[2.8, 11.30, 0.55, 6.39, 45.56,8.04,4.82,277.44,13.25,269.83,21.75]])


# In[38]:


#With this second scenario, the model predicts that the median value of owner-occupied homes 
#in 1000 dollars in the above suburb should be around 11,300


# # Step 9: Evaluate the model using the 10-fold cross-validation

# In[39]:


# Evaluate the algorithm
# Specify the K-size in this case 10-fold
num_folds = 10
#We must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 10
# Split the whole data set into folds
kfold = KFold(n_splits=num_folds, random_state=seed)
# For Linear regression, we can use MSE (mean squared error) value
# to evaluate the model/algorithm
scoring = 'neg_mean_squared_error'
# Train the model and run K-foLd cross-validation to validate/evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# Print out the evaluation results
# Result: the average of all the results obtained from the k-foLd cross-validation
print(results.mean())


# In[41]:


#After we train we evaluate Use K-Fold to determine if the model is acceptable 
#We pass the whole set because the system will divide for us -1.177 average
#of all errors (mean of square errors) 


# In[42]:


#


# In[ ]:




