import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";") #This is importing the data set and then seperating by the semi colon
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #Filtering the data into what I want to feed to the agent

predict = "G3" #The data we are training the model to find


x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression() #the actual model

linear.fit (x_train, y_train)
acc = linear.score(x_test, y_test) #accuracy test
print(acc) #show accuracy

print('Coefficient:  \n', linear.coef_)
print('Intercept \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])