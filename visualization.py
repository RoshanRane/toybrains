import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw

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
    
def show_images2(img_files, n_rows=1):
    fig, axes = plt.subplots(n_rows, len(img_files) // n_rows, figsize=(20, 3 * n_rows))
    for i, ax in enumerate(axes.flat):
        if i < len(img_files):
            img = mpimg.imread(img_files[i])
            # img = Image.open(img_files[i])
            ax.imshow(img)
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def show_table(df):
    # select only columns that start with 'gen_'
    gens = df.filter(regex='^gen_', axis=1)
    
    # sort dataframe cols by dtypes
    cols = gens.filter(like='gen_').dtypes.sort_values().index
    
    # set colors for continuous and discrete variables
    colors = ['blue'] + ['C'+str(i) for i in range(len(cols)-1)]
    
    f, axes = plt.subplots(4, 5, figsize=(20, 12), sharey=True)
    axes = axes.ravel()
    f.suptitle("Generative Variables used to generate the Images:", fontsize=16)
    f.supylabel("Count")
    
    for i, col in enumerate(cols):
        ax = axes[i]
        value = gens[col].value_counts()
        if len(value) > 10:
            gens[col].value_counts().plot.hist(ax=ax, bins=5, title=col.replace('gen_', ''), color=colors[i])
            ax.legend(['continuous'], loc='upper right')
        else:
            gens[col].value_counts().sort_index().plot.bar(ax=ax, title=col.replace('gen_', ''), color=colors[i])
            ax.legend(['discrete'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return gens

def generate_dict(gens):
    ''' generate the dictionary which contain subjectID according to Generative Variables
    '''
    # Initialize an empty dictionary
    subject_dict = {}
    
    # Loop through each column in the dataframe
    for col in gens.columns:
        unique_values = gens[col].unique()
        if len(unique_values) > 10:
            # Continuous case: divide into at least 4 groups
            groups = pd.cut(gens[col], bins=4, labels=False)
            for i in range(4):
                group_range = f'{gens[col].min() + i*(gens[col].max()-gens[col].min())/4:.2f}-{gens[col].min() + (i+1)*(gens[col].max()-gens[col].min())/4:.2f}'
                subject_ids = gens.index[groups == i].tolist()
                subject_dict[f'{col}_{group_range}'] = subject_ids
        else:
            # Discrete case: divide into groups based on unique values
            for value in unique_values:
                subject_ids = gens.index[gens[col] == value].tolist()
                subject_dict[f'{col}_{value}'] = subject_ids
    return subject_dict
    
def show_examples(gens, col = 0, DIR = "toybrains", N = 10):
    subject_dict = generate_dict(gens)
    keys = [key for key in subject_dict.keys() if gens.columns[col] in key]
    for key in keys:
        value = subject_dict[key][:N]
        img_files = [f'{DIR}/images/{i:05d}.jpg' for i in value]
        print(f'GV : {key}')
        show_images2(img_files=img_files, n_rows=1)
        print(f'{value}\n')