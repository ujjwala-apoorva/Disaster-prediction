#!/usr/bin/env python
# coding: utf-8

# # Data prediction for disaster
# 

# ## Importing Libraries 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Modelling
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[3]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[4]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB


# In[5]:


# KNeighbors
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


# Perceptron
from sklearn.linear_model import Perceptron


# In[7]:


# Support Vector Machines
from sklearn.svm import SVC


# In[8]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier


# In[9]:


# AdaBoost
from sklearn.ensemble import AdaBoostClassifier


# In[10]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[11]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier


# ## Loading of the Data

# Here, as the data sets have been taken from kaggel website, the data set was already divided into two sets. One could be used for the training of the models and the other for the test set

# In[13]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# ## Data exploration

# Here we would try to see the data set and understand it. Also the idea is to find the best features , unwanted or less important data sets or columns, so that we can later on do normalisation or data cleaning. Also dimension deduction if necessary. These steps are important to take to get a better prediction.

# In[14]:


train_data


# In[15]:


train_data.describe()


# In[16]:


print("Columns: \n{0} ".format(train_data.columns.tolist()))


# ## Data Preprocessing

# As mentioned above, after the data exploration, we saw that there could be some missing values in the columns , which could lead to wrong data prediction, or reduce the accuracy of the model.
# 
# So here we would be checking which columns have them, and then deal with them accordingly.

# ### Checking for the missing values

# In[17]:


missing_values = train_data.isna().any()
print('Columns which have missing values: \n{0}'.format(missing_values[missing_values == True].index.tolist()))


# In[18]:



print("Percentage of missing values in `Age` column: {0:.2f}".format(100.*(train_data.Age.isna().sum()/len(train_data))))
print("Percentage of missing values in `Cabin` column: {0:.2f}".format(100.*(train_data.Cabin.isna().sum()/len(train_data))))
print("Percentage of missing values in `Embarked` column: {0:.2f}".format(100.*(train_data.Embarked.isna().sum()/len(train_data))))


# The two ways of dealing with missing values could be, Either we remove those rows, with missing values or fill in those values with the mean average of the column.
# 
# Here as we have enough data sets, Instead of filling in the missing values, Removing them would be easier and better for the prediction.
# 
# 

# ### Checking for duplicates

# In[19]:


duplicates = train_data.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates))


# ### Checking for the Categorial Variables

# In[20]:


categorical = train_data.nunique().sort_values(ascending=True)
print('Categorical variables in train data: \n{0}'.format(categorical))


# ## Data Cleaning

# For the data cleaning as we saw in the data exploration part. We had many missing values for the "Cabin" column, and thus we would be dropping it.
# 
# The next set of columns we would be dropping would be the "name" , "Ticket", " Fare" and "Embarked" , as they would not be important for the data prediction

# In[21]:


def clean_data(data):
    # Too many missing values
    data.drop(['Cabin'], axis=1, inplace=True)
    
    # Probably will not provide some useful information
    data.drop(['Name', 'Ticket', 'Fare', 'Embarked'], axis=1, inplace=True)
    
    return data
    
train_data = clean_data(train_data)
test_data = clean_data(test_data)


# In[22]:


train_data.tail()


# ## Feature Engineering

# Although I have eliminated most of the columns for simplicity, in the future I am planning to recover those columns. They may contain some useful information.
# For now encoding the Sex column and filling Age column is enough to run a model.

# In[23]:


train_data['Sex'].replace({'male':0, 'female':1}, inplace=True)
test_data['Sex'].replace({'male':0, 'female':1}, inplace=True)

# Merge two data to get the average Age and fill the column
all_data = pd.concat([train_data, test_data])
average = all_data.Age.median()
print("Average Age: {0}".format(average))
train_data.fillna(value={'Age': average}, inplace=True)
test_data.fillna(value={'Age': average}, inplace=True)


# In[24]:


train_data.tail()


# ## Modeling

# Try different models with different parameters to understand which models give better results. Here in this section we will trying out different set of models and storing the data together. Then we will calculate the best model according to their accuracy, so that we can predict a disaster.
# 
# 

# In[25]:


# Set X and y
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']
test_X = test_data.drop(['PassengerId'], axis=1)


# In[26]:


# To store models created
best_models = {}

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def print_best_parameters(hyperparameters, best_parameters):
    value = "Best parameters: "
    for key in hyperparameters:
        value += str(key) + ": " + str(best_parameters[key]) + ", "
    if hyperparameters:
        print(value[:-2])
        
def get_best_model(estimator, hyperparameters, fit_params={}):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=estimator, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring="accuracy")
    best_model = grid_search.fit(train_X, train_y, **fit_params)
    best_parameters = best_model.best_estimator_.get_params()
    print_best_parameters(hyperparameters, best_parameters)
    return best_model

def evaluate_model(model, name):
    print("Accuracy score:", accuracy_score(train_y, model.predict(train_X)))
    best_models[name] = model


# In[27]:


print("Features: \n{0} ".format(X.columns.tolist()))


# ## Logistic Regression

# A logistic Regression model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.

# In[28]:


hyperparameters = {
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty' : ['l2'],
    'C'       : [100, 10, 1.0, 0.1, 0.01]
}
estimator = LogisticRegression(random_state=1)
best_model_logistic = get_best_model(estimator, hyperparameters)


# In[29]:


evaluate_model(best_model_logistic.best_estimator_, 'logistic')


# ## KNearestNeighbour

# In[30]:


hyperparameters = {
    'n_neighbors' : list(range(1,5)),
    'weights'     : ['uniform', 'distance'],
    'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size'   : list(range(1,10)),
    'p'           : [1,2]
}
estimator = KNeighborsClassifier()
best_model_kneighbors = get_best_model(estimator, hyperparameters)


# In[31]:


evaluate_model(best_model_kneighbors.best_estimator_, 'kneighbors')


# 88 % is quite a good score, and thus this model could be considered as a good fit model for this data set prediction.

# ## Perceptron

# In[32]:


hyperparameters = {
    'penalty'  : ['l1', 'l2', 'elasticnet'],
    'eta0'     : [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_iter' : list(range(50, 200, 50))
}
estimator = Perceptron(random_state=1)
best_model_perceptron = get_best_model(estimator, hyperparameters)


# In[33]:


evaluate_model(best_model_perceptron.best_estimator_, 'perceptron')


# Although perceptron is highly intelligent model, it is not a good fit for a data like this. As we do not get a very high score. So thus we would be moving on with a few more models to see the best score.

# ## Support Vector Machines

# In[34]:


hyperparameters = {
    'C'      : [0.1, 1, 10, 100],
    'gamma'  : [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel' : ['rbf']
}
estimator = SVC(random_state=1)
best_model_svc = get_best_model(estimator, hyperparameters)


# In[35]:


evaluate_model(best_model_svc.best_estimator_, 'svc')


# ## AdaBoost Classifier

# In[36]:


hyperparameters = {
    'n_estimators'  : [10, 50, 100, 500],
    'learning_rate' : [0.001, 0.01, 0.1, 1.0]
}
estimator = AdaBoostClassifier(random_state=1)
best_model_adaboost = get_best_model(estimator, hyperparameters)


# In[37]:


evaluate_model(best_model_adaboost.best_estimator_, 'adaboost')


# ## Decision Tree Classifier

# In[39]:


hyperparameters = {
    'criterion'         : ['gini', 'entropy'],
    'splitter'          : ['best', 'random'],
    'max_depth'         : [None, 1, 2, 3, 4, 5],
    'min_samples_split' : list(range(2,5)),
    'min_samples_leaf'  : list(range(1,5))
}
estimator = DecisionTreeClassifier(random_state=1)
best_model_decision_tree = get_best_model(estimator, hyperparameters)


# In[40]:


evaluate_model(best_model_decision_tree.best_estimator_, 'decision_tree')


# In[43]:


# Get predictions for each model and create submission files
for model in best_models:
    predictions = best_models[model].predict(test_X)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output


# As we had stored our models in a list, here we can see the prediction made by the best model.
# According to the data set and the multiple models we tried on, On seeing the accuracy, we can understand that K Nearest Neighbour would be the best match for this data set for data prediction. 
# 

# In[ ]:




