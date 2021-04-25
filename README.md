# IRUNet

![IRUNet architecture](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/IRUNet_network_architecture.png)<br /> <br />
## Original Dataset [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
![kaggle dataset description](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/kaggle_dataset_description.png)<br /> <br />
## Created [Datasets](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) for training and testing  <br />
![self created datasets](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/self_created_datasets_sample_image.png)<br /> <br />
## For model training <br />
### Histopathologic Cancer Detection dataset "train" was used to create the training set of images (clean,noise) named "noise_0_to_50" using "multipleImageNoiseCreator.py" program from Utils folder, meaning with noise ranges σ[0,50]
### Histopathologic Cancer Detection dataset "test" was used to create three testing set of images (clean,noise) named: noise_10, noise_25 and noise_50 using "multipleImageNoiseCreator.py" from Utils folder with a fixed noise. Fixed noise of σ = 10, σ = 25 , σ = 50
noise_0_to_50 folder is used for training <br />
## Sample images from the Histopathologic Cancer Detection dataset
![Kaggle dataset Sample Images](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/sample_images.png)<br /> <br />

## Denoising results
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised.png)<br /> <br />
