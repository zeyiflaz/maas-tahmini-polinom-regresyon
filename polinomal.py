# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 08:47:13 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri onisleme

# veri yukleme

veriler = pd.read_csv('maaslar.csv')
print(veriler)

# data drame dilimleme(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
#data da sorun çikarmasin diye dataframe olarak degisken verme
#numPY dizi (array) donusumu
X = x.values
Y = y.values

#lineer regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
#gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

#polynomial regression
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures # herhangi bir sayiyiyi polynomial olarak gostermeyi sagliyor
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
# gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color ='blue')
plt.show()

# 4. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures # herhangi bir sayiyiyi polynomial olarak gostermeyi sagliyor
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
# gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color ='blue')
plt.show()

# 6. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures # herhangi bir sayiyiyi polynomial olarak gostermeyi sagliyor
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color ='blue')
plt.show()

#tahminler
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))
