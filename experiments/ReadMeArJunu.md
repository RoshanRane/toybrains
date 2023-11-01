## Instructions for @arjun

- to run on CPU in dev mode     : `$python3 fit_DL_model.py -d -e 2 --data_dir dataset/toybrains_n10000`
    - This runs for me within 1 minute on GPU. 
    - Try different batch sizes with `-b` flag. Currently it is 128 but for running on the CPU, lower values like 8,16 or 32 might be relatively faster.
    - The deeprepvizlog is generated under `log/toybrains-[datasetname]/version_[latest]/deeprepvizlog`
        - since you run only 2 epochs there should be 2 checkpoint folders for each epoch in which I save the `tensors.tsv` (representations), `metadata.tsv` (labels, predicted labels, IDs), and `metrics.yaml` which is just stuff like accuracy and loss of the model at that epoch.
        - The csv file that can be uploaded to v1 should be present in this directory as `DeepRepViz-v1-[modelname].csv`
        - To see the tensorboard's rendering of these logs you can run `tensorboard --logdir ./` in the parent directory `log/toybrains-[datasetname]/version_[latest]/`
(doesnt work so well if run from outside)
            - it should fire up on http://localhost:6006/ 
            - on the top right menu bar select 'projector'
            - turn off 'Sphereize data' and choose color by 'labels'. 
    - Now the representation space should look the same between v1 and tensorboard since in the v1 table I use PCA. In the bottom left of tensorboard you can also use other complex dimensionality reduction methods such as TSNE and UMAP on the go.
    - Currently I don't provide all the variables to the tensorboard that I provide to DeepRepViz v1 table. I can do it if required.

- to run on a different dataset : `$python3 fit_DL_model.py -d -e 2 --data_dir dataset/toybrains_n10000_lowsignal`
    - you can generate additional datasets with the ../create_toybrains.py. Check the 0_tutorial.ipynb for guidance
    