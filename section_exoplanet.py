# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:20:02 2018

@author: Atharv
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl


prop = ['pl_orbper','pl_orbsmax','pl_orbeccen',
          'pl_bmassj','pl_radj','st_mass', 'st_teff', 'st_rad', 'st_metfe']

df1 = pd.read_csv('planets.csv', skiprows = 358)

def originalDataPlot(x,y,c,clabel,xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(300,210))
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


df1_pair = df1.loc[:, prop]
sns.pairplot(df1_pair.dropna())

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

