# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 10:19:00 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data3=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\22-07-15\15th\mlr\50_Startups.csv')

x=data3.iloc[:, :-1]
y=data3.iloc[:, 4]

x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)

#data preprocessing ends here

#now we start to fit and train model using liner regression

#fitting multiple liner regression to traning set
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

import statsmodels.formula.api as sm

x=np.append(arr=np.ones((50,1)).astype(	int),values=x,axis=1)


import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]

#ordnary least squared

regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#backward elimination delete x4

x_opt=x[:,[0,1,2,3,5]]

regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#delete 5

x_opt=x[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#delete 2

x_opt=x[:,[0,1,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()