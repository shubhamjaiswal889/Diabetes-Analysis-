#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import os
for dirname, _, filenames in os.walk(r'C:\Users\shubham.kj\Downloads\diabetes.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")  


# In[6]:


df = pd.read_csv(r'C:\Users\shubham.kj\Downloads\diabetes.csv')
df.head()


# In[7]:


df.info()


# In[8]:


# Creating the target and the features column and splitting the dataset into test and train set.

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Permutation Importance is calculated using the ELI5 library. ELI5 is a Python library which allows to visualize and debug various Machine Learning models using unified API. It has built-in support for several ML frameworks and provides a way to explain black-box models.

# In[10]:


# Training and fitting a Random Forest Model
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(X_train, y_train)


# In[12]:


pip install eli5


# In[13]:


# Calculating and Displaying importance using the eli5 library
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(X_test,y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# 
# 
# Interpretation
# 
#  1.  The features at the top are most important and at the bottom, the least. For this example, 'Glucose level' is the most important feature which decides whether a person will have diabetes, which also makes sense.
#  2. The number after the ± measures how performance varied from one-reshuffling to the next.
#  3. Some weights are negative. This is because in those cases predictions on the shuffled data were found to be more   accurate than the real data.
# 
# 

# # Partial Dependability

# In[14]:


# training and fitting a Decision Tree
from sklearn.tree import DecisionTreeClassifier
feature_names = [i for i in df.columns]
tree_model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)


# In[15]:


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[18]:


pip install graphviz


# In[19]:


from sklearn import tree
import graphviz


# In[20]:


tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names,filled = True)


# In[22]:


pip install graphviz


# In[27]:


pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature='Insulin')

# plot it
pdp.pdp_plot(pdp_goals, 'Insulin')
plt.show()


# In[28]:


features_to_plot = ['Glucose','Insulin']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=X_test, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)
plt.show()


# In[29]:


row_to_show = 10
data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

tree_model.predict_proba(data_for_prediction_array)


# SHAP values are calculated using the Shap library which can be installed easily from PyPI or conda. SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we’d make if that feature took some baseline value.

# In[30]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(tree_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)


# In[31]:


pip install shap


# In[32]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(tree_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)


# In[33]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# 
# 
# Interpretation
# 
# The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue
# 
#     The base_value here is 0.3576 while our predicted value is 1.
#     Glucose = 158 has the biggest impact on increasing the prediction, while
#     Age feature has the biggest effect in decreasing the prediction.
# 
# 

# SHAP Summary Plots

# In[34]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(tree_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_test)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1],X_test)


# 
# 
# For every dot:
# 
#     Vertical location shows what feature it is depicting
#     Color shows whether that feature was high or low for that row of the dataset
#     Horizontal location shows whether the effect of that value caused a higher or lower prediction.
# 
# The point in the upper left was depicts a person whose glucose level is less thereby reducing the prediction of diabetes by 0.4.
# 

# In[ ]:




