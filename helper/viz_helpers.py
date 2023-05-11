# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.image as mpimg
from sklearn import datasets, linear_model
from tqdm.notebook import tqdm
import random
import math
import json
from  matplotlib.ticker import FuncFormatter
from matplotlib.colors import is_color_like
from colordict import ColorDict, rgb_to_hex


def show_images(img_files, n_rows=1):
    assert isinstance(img_files[0], (str,np.ndarray, np.generic)), "img_files \
should either me a string path to the image files or numpy arrays"   
    n_cols = 10
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols,n_rows), 
                           sharex=True, sharey=True)
    # f.suptitle("Toy brains dataset:")
    axes = axes.ravel()

    for i, img in enumerate(img_files):
        if i<len(axes):
            if isinstance(img, str):
                img = mpimg.imread(img)
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_col_dists(df, title=''):
    
    # convert column to appropriate dtype and then decide the plottype
    df_copy = df.reindex(sorted(df.columns), axis=1).copy()
    cols = df_copy.columns 

    plottypes = {}
    for col in cols:
        if df_copy[col].nunique()==2:
            plottypes.update({col:'pie'})
        elif df_copy[col].dtype.name == 'object' or isinstance(df_copy.loc[0,col], str):
            plottypes.update({col:'bar'})
        elif 'float' in df_copy[col].dtype.name:
            # round the floats to 2 decimal
            df_copy[col] = df[col].round(1)
            # use hist plot if there are more than 10 states  
            if df_copy[col].nunique()<10:
                plottypes.update({col:'bar'})
            else:
                plottypes.update({col:'hist'})
            # if all values are int then covert col to int dtype
            if df_copy[col].dropna().apply(float.is_integer).all():
                df_copy[col] = df_copy[col].astype(int)
            
    # re-sort dataframe cols by dtypes
    cols = df_copy.dtypes.sort_values().index 
    # decide the number of rows and columns in the plot
    subplot_ncols = 5 if len(cols)>=5 else len(cols)
    subplot_nrows = len(cols)//subplot_ncols
    subplot_overflows = len(cols)%subplot_ncols
    if subplot_overflows!=0: subplot_nrows+=1
    # create subplots set attributes
    f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                          figsize=(4*subplot_ncols,3*subplot_nrows))
    if title: f.suptitle(title, fontsize=16)
    f.supylabel("Count")
    
    for i, ax in enumerate(axes.ravel()):
        col = cols[i]
        plottype = plottypes[col]
        # print('[D]',col, plottypes[col])
        if i >= subplot_ncols*(len(cols)//subplot_ncols) + subplot_overflows:
            ax.axis('off')
            
        elif plottype == 'bar':
            # check if the attribute represents colors then use the same color names for the bar plot
            if isinstance(df_copy.loc[0,col], str) and is_color_like(df_copy.loc[0,col].split('-')[-1]):
                colormap = ColorDict()
                colors = [rgb_to_hex(colormap[c.split('-')[-1]]) for c in df_copy[col].sort_values().unique().tolist()]

                sns.countplot(data=df_copy, x=col, ax=ax, palette=colors,
                             order=df_copy[col].sort_values().unique())
            else:
                sns.countplot(data=df_copy, x=col, ax=ax)
            # format the xtick labels 
            if 'int' in df_copy[col].dtype.name:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            elif isinstance(df_copy.loc[0,col], str):
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                
        elif plottype == 'hist':
            bins = 10 if df_copy[col].nunique()>10 else df_copy[col].nunique()
            sns.histplot(data=df_copy, x=col, bins=, ax=ax)
        elif plottype == 'pie':
            cnt = df_copy[col].value_counts().sort_index()
            ax.pie(cnt, labels=cnt.index,
                    colors=sns.color_palette('pastel'), autopct='%.0f%%')
            
        ax.set_title(col)
        ax.set_xlabel(None)

    plt.tight_layout()