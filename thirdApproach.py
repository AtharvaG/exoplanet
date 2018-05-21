# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:55 2018

@author: Atharv
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
import matplotlib as mpl




df = pd.read_csv('planets.csv', skiprows = 358)  #skips first 358 rows, these are garbage rows      

df = df[df['pl_pnum'] == 1]    



prop = ['pl_orbper','pl_orbsmax','pl_orbeccen',
          'pl_bmassj','pl_radj','st_mass', 'st_teff', 'st_rad', 'st_metfe']
df = df.loc[:, prop]
df = df.dropna(subset = prop, how = 'any', axis = 0)

pl_prop = ['pl_orbper','pl_orbsmax','pl_orbeccen',
          'pl_bmassj','pl_radj']
X = df.loc[:, pl_prop]
X = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(X, columns = pl_prop)

st_prop = ['st_mass', 'st_teff', 'st_rad', 'st_metfe']
y = df.loc[:, st_prop] 
y = StandardScaler().fit_transform(y.values)      
y = pd.DataFrame(y, columns = st_prop ) 

y.isnull().sum()


sns.regplot(df['st_mass'], df['pl_bmassj'])



sns.set(font='serif', font_scale=1.4, style='ticks')
palette = sns.hls_palette(8, l=.3, s=.8)
pal = palette.as_hex()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(y,X,train_size = 0.8)



from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=3)
pls2.fit(X_train, y_train)

y_pred = pls2.predict(X_test)

from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)


# Histogram of Eccentricity
fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df['pl_orbeccen'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel('Eccentricity')
plt.ylabel('Number of Planets')
#ax.set_xscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df['pl_orbsmax'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Semi Major Axis (AU)')
plt.ylabel('Number of Planets')
#ax.set_xscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df['pl_orbper'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Orbital Period (Days)')
plt.ylabel('Number of Planets')
#ax.set_xscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df['st_metfe'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Metalicity (dex)')
plt.ylabel('Number of Planets')
#ax.set_xscale('log')
plt.legend()
plt.show() 

fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df['pl_radj'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Radius ($R_{Jup}$)')
plt.ylabel('Number of Planets')
ax.set_xscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
sns.distplot(df1['st_lum'].dropna(),
             kde=False, rug=False, #color='darkcyan',
             hist_kws={"edgecolor": "k", "linewidth": 1.5, "alpha": 0.5})
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Luminosity [log(soalr)]')
plt.ylabel('Number of Planets')
#ax.set_xscale('log')
plt.show()


fig, ax = plt.subplots(figsize=(10,7))


sns.regplot(df['pl_orbeccen'],df['pl_orbsmax'])
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Eccentricity')
plt.ylabel(r'Semi Major Axis (AU)')
#ax.set_xscale('log')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(df['st_metfe'],df['pl_orbper'], c = df['pl_orbeccen'], cmap = palette)
fig.colorbar(points)
sns.regplot(df['st_metfe'],df['pl_orbper'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Metalicity (dex)')
plt.ylabel(r'Orbital Period (Days)')
#ax.set_xscale('log')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(y['st_mass'],X['pl_bmassj'], c = X['pl_orbsmax'], cmap = palette)
fig.colorbar(points)
sns.regplot(y['st_mass'],X['pl_bmassj'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Stellar Mass ')
plt.ylabel(r'Planetary Mass')
#ax.set_xscale('log')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(y['st_mass'],X['pl_radj'], c = X['pl_orbper'], cmap = palette)
fig.colorbar(points)
sns.regplot(y['st_mass'],X['pl_radj'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Stellar Mass ')
plt.ylabel(r'Planetary Radius')
#ax.set_xscale('log')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(df['st_teff'],df['pl_radj'], c = df['pl_bmassj'], cmap = palette)
fig.colorbar(points)
#sns.regplot(y['st_teff'],X['pl_radj'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Temp ')
plt.ylabel(r'Planetary Radius')
#ax.set_xscale('log')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(df['pl_orbsmax'],df['st_teff'], c = df['pl_orbper'], cmap = palette)
fig.colorbar(points)
#sns.regplot(y['st_teff'],X['pl_radj'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Semi Major Axis (AU) ')
plt.ylabel(r'Effective Temperature (K)')
#ax.set_xscale('log')
ax.legend()
plt.show()


fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette( as_cmap = True)
points = plt.scatter(df['pl_bmassj'],df['pl_radj'], c = df['pl_orbeccen'], cmap = palette)
fig.colorbar(points)
#sns.regplot(y['st_teff'],X['pl_radj'], scatter=False)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plt.xlabel(r'Planetary Mass $(M_{jup})$')
plt.ylabel(r'Planetary Radius $(R_{jup})$')
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()


def plotScatter(x,y,c,xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10,7))
    palette = sns.cubehelix_palette( as_cmap = True)
    points = plt.scatter(df[x],df[y], c = df[c], cmap = palette)
    fig.colorbar(points)
    #sns.regplot(y['st_teff'],X['pl_radj'], scatter=False)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()


plotScatter('st_mass', 'st_rad', 'st_teff', r'Stellar Mass $(M_{s})$', r'Stellar Radius $(R_{s})$' )
plotScatter('st_mass', 'st_rad', 'st_metfe', r'Stellar Mass $(M_{s})$', r'Stellar Radius $(R_{s})$' )
plotScatter('pl_bmassj', 'pl_radj', 'pl_orbeccen', r'Planetary Mass $(M_{jup})$', r'Planetary Radius $(R_{jup})$' )


df1 = pd.read_csv('planets.csv', skiprows = 358)

def originalDataPlot(x,y,c,clabel,xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(200,140))
    palette = sns.cubehelix_palette( as_cmap = True)
    points = plt.scatter(df1[x],df1[y], c = df1[c], cmap = palette)
# =============================================================================
#                          norm= mpl.colors.LogNorm(
#             vmin =df1[c].min(), vmax = df1[c].max()))
# =============================================================================
    g = fig.colorbar(points)
    g.ax.set_ylabel(clabel)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

originalDataPlot('pl_orbsmax' , 'st_teff' , 'pl_orbeccen', 'Orbital Eccentricity',r'Semi Major Axis (AU)', r'Effective Temp (K)')
originalDataPlot('st_mass' , 'pl_bmassj' , 'st_lum','Luminosity (log(solar))',r'Stellar Mass M$_{\bigodot}$', r'Planetary Mass M$_{jup}$')
originalDataPlot('pl_orbper', 'pl_radj', 'pl_orbsmax','Semi Major (AU)', r'Orbital Period (Days)', r'Radius of Planets R$_{jup}$')
originalDataPlot('pl_bmassj', 'pl_radj', 'pl_orbper','Orbital Period (Days)', r'Planetary Mass $(M_{jup})$', r'Planetary Radius $(R_{jup})$')
originalDataPlot( 'st_teff','st_mass', 'st_metfe',  'Stellar Metallicity',  r'Effective Temp (K)',r'Stellar Mass M$_{\bigodot}$',)
originalDataPlot('st_metfe','pl_bmassj', 'pl_orbsmax', 'Semi Major (AU)', 'Stellar Metallicity[Fe/H]',r'Planetary Mass $(M_{jup})$', )

df2 = df1[df1['pl_pnum'] == 1]

def onePlanet(x,y,c,clabel,xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(200,140))
    palette = sns.cubehelix_palette( as_cmap = True)
    points = plt.scatter(df2[x],df2[y], c = df2[c], cmap = palette)
# =============================================================================
#                          norm= mpl.colors.LogNorm(
#             vmin =df1[c].min(), vmax = df1[c].max()))
# =============================================================================
    g = fig.colorbar(points)
    g.ax.set_ylabel(clabel)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

onePlanet('pl_orbsmax' , 'st_teff' , 'pl_orbeccen', 'Orbital Eccentricity',r'Semi Major Axis (AU)', r'Effective Temp (K)')
onePlanet('st_mass' , 'pl_bmassj' , 'st_lum','Luminosity (log(solar))',r'Stellar Mass M$_{\bigodot}$', r'Planetary Mass M$_{jup}$')
onePlanet('pl_orbper', 'pl_radj', 'pl_orbsmax','Semi Major (AU)', r'Orbital Period (Days)', r'Radius of Planets R$_{jup}$')
onePlanet('pl_bmassj', 'pl_radj', 'pl_orbper','Orbital Period (Days)', r'Planetary Mass $(M_{jup})$', r'Planetary Radius $(R_{jup})$')
onePlanet('st_teff', 'st_mass', 'st_metfe', 'Stellar Metallicity',  r'Effective Temp (K)',r'Stellar Mass M$_{\bigodot}$')
onePlanet( 'st_metfe', 'pl_bmassj','pl_orbsmax', 'Semi Major (AU)', 'Stellar Metallicity[Fe/H]',r'Planetary Mass $(M_{jup})$' )




def allPlanet(x,y,c,clabel,xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(200,140))
    palette = sns.cubehelix_palette( as_cmap = True)
    points = plt.scatter(df2[df2['pl_pnum'] == 1][x],df2[df2['pl_pnum'] == 1][y], c = df2[df2['pl_pnum'] == 1][c],
                         cmap = palette, marker='s')
# =============================================================================
#                          norm= mpl.colors.LogNorm(
#             vmin =df1[c].min(), vmax = df1[c].max()))
# =============================================================================
    g = fig.colorbar(points)
    g.ax.set_ylabel(clabel)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

sns.pairplot(df, vars= prop[2:])
#fig, ax = plt.subplots()
ax = sns.pairplot(df,vars = prop[:-4])
ax.set()
plt.show()
sns.pairplot(df1, vars = prop[-4:]) 
df1_pair = df1.loc[:, prop]
sns.pairplot(df1_pair.dropna())

"""
observations 
a shift in eccentricity is not observable as most of the planets have eccentricity < 0.2 

a shift in semi major axis can give some variation, let's see , nope

TODO:
    -learn NN and run it on mss v radius relation (sounds weird)
    -read the papers on met of planets and the one on planet formation.
    -from all the graphs plotted write down observation in a notebook,
     create a table if possible
    -document every other thing from the beginning before it gets too late
    -
    

"""

