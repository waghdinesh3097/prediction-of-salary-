# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,:2]
postion=pd.get_dummies(x['Position'])
x.drop(['Position'],axis=1,inplace=True)
X=pd.concat([x,postion],axis=1)
X.shape

y=dataset.iloc[:,2]
y.shape

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl2','wb'))

model = pickle.load(open('model.pkl2','rb'))
print(model.predict([[2,9,1]]))



#y = dataset.iloc[:, -1]
#print(y)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#Fitting model with trainig data
#regressor.fit(X, y)

# Saving model to disk
#pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))