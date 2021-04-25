# IRUNet

![IRUNet architecture](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/IRUNet_network_architecture.png)<br /> <br />
## Original Dataset [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
![kaggle dataset description](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/kaggle_dataset_description.png)<br /> <br />
## Created [Datasets](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) for training and testing  <br />
![self created datasets](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/self_created_datasets_sample_image.png)<br /> <br />
## For model training <br />
#### Histopathologic Cancer Detection dataset "train" was used to create the training set of images (clean,noise) named "noise_0_to_50" using "multipleImageNoiseCreator.py" program from Utils folder, meaning with noise ranges σ[0,50]
## For model testing <br />

#### Histopathologic Cancer Detection dataset "test" was used to create three testing set of images (clean,noise) named: noise_10, noise_25 and noise_50 using "multipleImageNoiseCreator.py" program from Utils folder with a fixed noise: σ = 10, σ = 25 , σ = 50.

## Sample images from the Histopathologic Cancer Detection dataset
![Kaggle dataset Sample Images](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/sample_images.png)<br /> <br />

## Denoising results
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised.png)<br /> <br />


## Instruction of use <br />
### Model Training <br />
Make sure you have downloaded and extracted the files of the training dataset "noise_0_to_50" from the drive folder which is ready to use. <br />

Alternatively you may download the original [dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) from Kaggle and extract the files <br /> <br />
After downloading the original dataset from Kaggle you may need to: <br /><br />
     I) Use the snippet "renaming_files_ascending_order" located in "Utils" folder <br />
    II) Use the program "multipleImageNoiseCreator.py" located at "Utils" <br /><br />
At this point you should have the dataset ready to use for training. <br />
We may now configure the "main.py" program located in "Code" folder.
