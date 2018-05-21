# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:27:21 2018

@author: Atharv
"""

dt_X = dt_X.sort_values('pl_orbper') #sorts values 
dt_X['pl_orbper'].isnull().sum()

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

x = 10 ** np.arange(1, 10)
y = 10 ** np.arange(1, 10) * 2

df1 = pd.DataFrame(data=y, index=x)

df1.plot(loglog=True, legend=False)

plt.show()


plt.figure(figsize = (18,15))
plt.scatter(dt_std_df['pl_orbsmax'], dt_std_df['pl_orbper'])
plt.title('Scatter Plot')
plt.xlabel('Semi Major Axis of the orbit of planet')
plt.ylabel('Orbital period of planet')
plt.show()
plt.scatter(dt_std_df['pl_orbsmax']**3, dt_std_df['pl_orbper']**2)
plt.title('Scatter Plot')
plt.xlabel('R^3')
plt.ylabel('T^2')
plt.show()
plt.scatter(dt_std_df['pl_radj'], dt_std_df['pl_orbper'])
plt.title('Scatter Plot')
plt.xlabel('Radius of Planet')
plt.ylabel('Orbital Period of Planet')
plt.show()
plt.scatter(dt_std_df['pl_orbsmax'], dt_std_df['st_teff'])
plt.title('Scatter Plot')
plt.xlabel('Orbital Period of Planet')
plt.ylabel('Effective Temp of Star')
plt.show()













