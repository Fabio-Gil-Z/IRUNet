# IRUNet

![IRUNet architecture](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/IRUNet_network_architecture.png)<br /> <br />
## Original Dataset [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
![kaggle dataset description](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/kaggle_dataset_description.png)<br /> <br />
## Created [Datasets](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) for Training and Testing <br />
![self created datasets](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/self_created_datasets_sample_image.png)<br /> <br />
## For model training <br />
#### [Histopathologic Cancer Detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) "train" was used to create the training set of images (clean,noise) named [noise_0_to_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) using [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) program from [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder, meaning with noise ranges between σ[0,50].

## For model testing <br />
#### [Histopathologic Cancer Detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) "test" was used to create three testing set of images (clean,noise) named: [noise_10](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing), [noise_25](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) and [noise_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) using [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) program from [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder with a fixed noise: <br /> σ = 10, σ = 25 , σ = 50.

## Sample images from the Histopathologic Cancer Detection dataset
![Kaggle dataset Sample Images](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/sample_images.png)<br /> <br />

## Denoising results
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised.png)<br /> <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised_2.png)<br /> <br />

## Requirements <br />
### [Tensorflow](https://www.tensorflow.org/install) 2.0 or greater
### [Cuda](https://developer.nvidia.com/cuda-toolkit-archive) and [Cudnn](https://developer.nvidia.com/cudnn) 10.1 or better
### [Python3](https://www.python.org/downloads/)
### This work has been developed with: <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/pc_specs.png)<br /> <br />

## Instructions of use <br />
### Model Training <br />
Make sure you have downloaded and extracted the files of the training dataset [noise_0_to_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) from the drive folder which is ready to use. <br />

Alternatively you may download the original [dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) from Kaggle and extract the files. <br /> <br />
If you downloaded the original dataset from Kaggle you may need to: <br /><br />
     I) Use the snippet [renaming_files_ascending_order](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/ubuntu_snippets/renaming_files_ascending_order) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder. <br />
    II) Use the program [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder. <br />
   III) Please make sure you have a folder with name files 1_clean.tif, 1_noise.tif, 2_clean.tif, 2_noise.tif ... etc. <br /><br />
   [Here](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/testSample_10) is an example how your folder should look like with only 10 images.
At this point you should have the dataset ready to use for training. <br /><br />
We may now configure the "main.py" program located in "Code" folder. <br />
Default settings for training: <br /><br />
**BATCH_SIZE = 32** <br />

**DATASET_DIRECTOY = "path/to/noise_0_to_50"** <br />
Here you may type the dataset directory in which you downloaded / created the training dataset. <br />


**DIRECTORY_TO_SAVE_MODELS = "Models"** <br />
Default name: Models <br />


**DIRECTORY_TO_SAVE_TENSORBOARD = "Tensorboard"** <br />
Default name: Tensorboard <br />


**DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT = "Epoch_results"** <br />
Default name: Epoch_results <br />
*In this folder images will be saved after each epoch, showcasing the learning progress of the network.* <br />


**modelName = "myDenoiser"** <br />
**weights = "myDenoiser"**   <br />
*Make sure both of these names match, for more information look at "main.py".* <br />


**loadWeights = False**  <br />
Defaults to "False" <br />
*Use in case you want to resume your training if for some reason it was stopped, you may want to change it to **"True"**.* <br />

**epoch_assignation = 1000** <br />
*You may choose a number of epochs for training.* <br />

**filters = 16** <br />
*You may choose to change the number of filters* <br />


**optimizer = ADAM_optimizer** <br />
Defaults to: ADAM_optimizer <br />
*You may choose other optimizers, for more information look at "main.py"*.<br />


**loss_function   = "MeanAbsoluteError"** <br />
Defaults to: "MeanAbsoluteError" <br />
*You may try: "MeanSquaredError"*.

We have successfully finished configuring our "main.py" file. <br />

You may run the program with "python3 main.py" execution line at the terminal. <br />

### Model Testing <br />

#### Testing over a group of images ( average testing PSRN / SSIM )
In order to test the model over a group of images, we will be using [averageTester.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/averageTester.py) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder.

#### Example of expected output
![averageTester_expected_output](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/averageTester_expected_output.png)<br /> <br />

#### Testing over a single image
