# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:56:53 2021

@author: rmeredith
"""
# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##import plotly.express as px
##import hvplot.pandas
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import database from CSV
toc_db = pd.read_csv('Multivariant_Regression_TOC.csv')

# Pull out the dependent variables from the "Redox and Production Proxies" to use in model (i.e. 'Mo','U','Ni','Cu','Cr','Zn','Pb','Th','V','Ba','Se','S','Fe')
# Remove elements that are not present in dataset (i.e. if sum of element equals zero)
x = pd.DataFrame(toc_db[['Mo','U','Ni','Cu','Cr','Zn','Pb','Th','V','Ba','Se','S','Fe']])
# Replace nan with zero's
x = x.fillna(0.0001)

# Pull out the independent variable "TOC"
y = pd.DataFrame(toc_db[['TOC']])

# Remove features that may have variance that are orders of magnitude larger than others. 
# StandardScaler sets whole data set to Zero (need to investigate if each element is 0 to 1)
##x_scaled = StandardScaler().fit_transform(x)
x_scaled = MinMaxScaler().fit_transform(x)

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = .15, random_state = 42)

# Apply Linear Regression Model to split data (Intercept is set to Zero)
model = lr().fit(x_train, y_train)
##model = lr(fit_intercept=False).fit(x_train, y_train)

# Getting "predict" from LR Model
y_predict = model.predict(x_test)

# Convert y_predict to DataFrame
y_predict_df = pd.DataFrame(y_predict)

# Finding R-squared of LR Model
score = r2_score(y_test,y_predict)

# Finding Coefficient & Y-Intercept of LR Model (Un-Null 'intercept' code if LR model has intercept)
coef = model.coef_
coef_df = pd.DataFrame(coef, columns=x.columns)
intercept = model.intercept_
intercept_df = pd.DataFrame(intercept)
intercept_df.columns = ['Intercept']

# Merging coef and intercept for export
C_I = pd.concat([coef_df,intercept_df], axis=1).T


# Write Coefficients & Y-intercept to Excel
df = pd.DataFrame(C_I).T
##df = pd.DataFrame(coef_df)
df.to_excel(excel_writer = "TOC_Coefficient.xlsx", index=False)

# Scatter Plot of Prediction vs TOC
plt.style.use("seaborn-darkgrid")
plt.scatter(x=y_predict_df,y=y_test,c=y_predict_df,marker="d",cmap="jet", edgecolor="black")
plt.title("TOC vs Predicted TOC")
plt.xlabel("Predicted TOC")
plt.ylabel("TOC (wt%)")
plt.legend([f'Rsq ={score: .2f}'],loc="best",shadow=True,frameon=True,fancybox=True)
##plt.colorbar()

# Plot a trendline
m, b = np.polyfit(y_predict_df[0], y_test, 1)
plt.plot(y_test,m*y_test + b, 'k--', label='linear')

# Save plot as PNG
plt.savefig('TOC_Scatter.png', bbox_inches='tight')

# Show the Scatter Plot
plt.show()