The following link sends you to the datasets in my drive
## Created [Datasets](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) for Training and Testing <br />
![self created datasets](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/self_created_datasets_sample_image.png)<br /> <br />

The datasets noise_10, noise_25, noise_50 are designed for testing.
These are datasets with image pairs (clean,noise) with fixed noise 10,25,50 depending on the folder.

The folder noise_0_to_50 is the training dataset I used for training.
It was corrupted with noise [0,50] of standard deviation.

A total of 4400 images of each standard deviation were created with my python program
[multipleImageNoiseCreator](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) in the [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder.

It is poissible to use it to create your own training dataset.


Remember this is the training kaggle dataset:
## Original Dataset [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
![kaggle dataset description](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/kaggle_dataset_description.png)<br /> <br />


