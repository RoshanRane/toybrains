import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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