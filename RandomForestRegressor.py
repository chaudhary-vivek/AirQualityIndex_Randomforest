# In this program we will use random forest to predict the AQI

# Part 1: Preprocessing

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df=pd.read_csv('D:\Docs\DS\AQI\Data\Real-Data/Real_Combine.csv')
# Checking for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Since there are very few null values, we can drop them
df = df.dropna()
# Defining dependent and independent features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Seeing the feature importance using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)

# Plotting the importance using ExtraTreesRegressor
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Splitting the data using train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Part 2: Making random froest

# Importing random forest
from sklearn.ensemble import RandomForestRegressor

# Fitting the data into the random forest
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

# Printing the coeficient of determination for training and tsting datasets
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
# Training set Rsguare is 0.97
# Testing set RMS is 0.79 which makes it not very accurate

# Importing the cross validation score
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()
# Cross validation score is 0.73

# Part 3: Prediction

# Making prediction
prediction=regressor.predict(X_test)
# Plotting the difference of the predicted avlue and the test value
sns.distplot(y_test-prediction)
# The distribution looks normally distributed, therefore the predictions were good
plt.scatter(y_test,prediction)

# Part 4: Hyper Parameter tuning

# Initiaitng the regressor
RandomForestRegressor()

# Importing RandomizedSearchCV, this is faster than GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# This code will give 12 different numbers of decission trees between 100 and 1200
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# This will decide the number of features in each split
max_features = ['auto', 'sqrt']
# This will decide the maximum levels in the tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Now we create a dictionary of the different parameters which were specified above
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Initiating the random forest model to tune
rf = RandomForestRegressor()
# Doing random grid to search for best hyper parameters and do 3 fold cross validation
# Scoring is negative RMS
# Iterations are 100
# Random state is 42 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

# Fitting data in the model
rf_random.fit(X_train,y_train)

# Printing the best parameters
print('The best parameters are {}'.format(rf_random.best_params_))
# Printing the best score
print('The best score is {}'.format(rf_random.best_score_))
# The score is -1555, twice as better as linear regression or decision tree

# Part 5: Prediction and evaluation

# Making the prediction on test data
predictions=rf_random.predict(X_test)
# Plotting the difference of the predicted value and the test value
sns.distplot(y_test-predictions)
# The ditribution is normal, with much lesser kurtosis. Hence the model is good
# Plotting the predicted value versus test value
plt.scatter(y_test,prediction)

# Printning the MAE, MSE, RMSE
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Part 6: Dumping the data using pickle for future deployment
 
# Importing pickkle
import pickle

# opening the file to dump the model
file = open('random_forest_regression_model.pkl', 'wb')
# dumping the tuned model into the file
pickle.dump(rf_random, file)


