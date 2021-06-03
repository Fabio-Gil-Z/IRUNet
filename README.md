# Blind microscopy image denoising with a deep residual and multiscale encoder/decoder network
# Abstract
In computer-aided diagnosis (CAD) focused on microscopy, denoising improves the quality of image analysis. In general, the accuracy of this process may depend both on the experience of the microscopist and on the equipment sensitivity and specificity. A medical image could be corrupted by both intrinsic noise, due to the device limitations, and, by extrinsic signal perturbations during image acquisition. Nowadays, CAD deep learning applications pre-process images with image denoising models to reinforce learning and prediction. In this work, an innovative and lightweight deep multiscale convolutional encoder-decoder neural network is proposed. Specifically, the encoder uses deterministic mapping to map features into a hidden representation. Then, the latent representation is rebuilt to generate the reconstructed denoised image. Residual learning strategies are used to improve and accelerate the training process using skip connections in bridging across convolutional and deconvolutional layers. The proposed model reaches on average 38.38 of PSNR and 0.98 of SSIM on a test set of 57458 images overcoming state-of-the-art models in the same application domain.

# IRUNet - [Paper](https://arxiv.org/abs/2105.00273)
![IRUNet architecture](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/IRUNet_network_architecture.png)<br /> <br />
## Original Dataset [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
![kaggle dataset description](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/kaggle_dataset_description.png)<br /> <br />
## Created [Datasets](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) for Training and Testing <br />
![self created datasets](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/self_created_datasets_sample_image.png)<br /> <br />
## Cite this paper
@article{hernan2021blind, <br />
  title={Blind microscopy image denoising with a deep residual and multiscale encoder/decoder network}, <br />
  author={Fabio Hernán Gil Zuluaga and Francesco Bardozzo and Jorge Iván Ríos Patiño and Roberto Tagliaferri}, <br />
  journal={arXiv e-prints}, <br />
  pages={arXiv--2105}, <br />
  year={2021} <br />
} <br />

## For model training <br />
#### [Histopathologic Cancer Detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) "train" was used to create the training set of images (clean,noise) named [noise_0_to_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) using [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) program from [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder, meaning with noise ranges between σ[0,50].

## For model testing <br />
#### [Histopathologic Cancer Detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) "test" was used to create three testing sets of images (clean,noise) named: [noise_10](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing), [noise_25](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) and [noise_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) using [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) program from [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder with a fixed noise: <br /> σ = 10, σ = 25 , σ = 50.

## Sample images from the Histopathologic Cancer Detection dataset
![Kaggle dataset Sample Images](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/sample_images.png)<br /> <br />

## Denoising results
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised_2.png)<br /> <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/denoised_and_table_results.png) <br /> <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/test_results.png)<br /> <br />
## Following are the results obtained using the current IRUNet model, loading its weights and training it for 70 additional epochs.
It can be appreciated that the loss is not changing (Cyan), however, we can appreciate that the PSNR is indeed changing (Orange). This was the reason we used PSNR to measure the best model instead of loss. Additionally, the best model currently is the one obtained at epoch 65 which means the model can still be improved. <br /> <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/results_at_the_end_of_each_epoch.png)<br /> <br />


## Requirements <br />
### [Tensorflow](https://www.tensorflow.org/install) 2.0 or greater
### [Cuda](https://developer.nvidia.com/cuda-toolkit-archive) and [Cudnn](https://developer.nvidia.com/cudnn) 10.1 or better
### [Python3](https://www.python.org/downloads/)
### This work has been developed with: <br />
![Denoising results](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/pc_specs.png)<br /> <br />
---
## Instructions of use <br />
### Model Training <br />
Make sure you have downloaded and extracted the files of the training dataset [noise_0_to_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) from the drive folder which is ready to use. <br />

Alternatively it is possible to download the original [dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data) from Kaggle and extract the files. <br /> <br />
If you downloaded the original dataset from Kaggle, please follow the next steps: <br /><br />
     I) Use the snippet [renaming_files_ascending_order](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/ubuntu_snippets/renaming_files_ascending_order) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder. <br />
    II) Use the program [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder. <br />
   III) Please make sure you have a folder with name files 1_clean.tif, 1_noise.tif, 2_clean.tif, 2_noise.tif ... etc. <br /><br />
   [Here](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/testSample_10) is an example how your folder should look like with only 10 images.<br />
## *At this point you should have the dataset ready to use for training.* <br /><br />
### We may now configure the [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py) program located in [Code](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code) folder. <br /><br />
### Following are the default settings of [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py) : <br /><br />
**BATCH_SIZE = 32** <br />

**DATASET_DIRECTOY = "path/to/noise_0_to_50"** <br />
Here it is possible to type the dataset directory in which you downloaded / created the training dataset. <br />


**DIRECTORY_TO_SAVE_MODELS = "Models"** <br />
Default name: Models <br />


**DIRECTORY_TO_SAVE_TENSORBOARD = "Tensorboard"** <br />
Default name: Tensorboard <br />


**DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT = "Epoch_results"** <br />
Default name: Epoch_results <br />
*In this folder images will be saved after each epoch, showcasing the learning progress of the network.* <br />


**modelName = "myDenoiser"** <br />
**weights = "myDenoiser"**   <br />
*Make sure both of these names match, for more information look at [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py).* <br />


**loadWeights = False**  <br />
Defaults to "False" <br />
*Use in case you want to resume your training if for some reason it was stopped, it is possible to change it to **"True"**.* <br />

**epoch_assignation = 1000** <br />
 it is possible to choose a number of epochs for training. <br />

**filters = 16** <br />
 it is possible to change the number of filters <br />


**optimizer = ADAM_optimizer** <br />
Defaults to: ADAM_optimizer <br />
 it is possible to choose other optimizers, for more information look at [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py).<br />


**loss_function   = "MeanAbsoluteError"** <br />
Defaults to: "MeanAbsoluteError" <br />
 it is possible to use: "MeanSquaredError".

### We have successfully finished configuring our [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py) file. <br />
 Now it should be possible to run the program with "python3 [main.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/main.py)" execution line at the terminal. <br />

### Model Testing <br />

#### Testing over a group of images ( average testing PSRN / SSIM )
In order to test the model over a group of images, we will be using [averageTester.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/averageTester.py) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder.

#### Example of expected output
![averageTester_expected_output](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/averageTester_expected_output.png)<br /> <br />

####  it is possible to configure the file to change the default path from the sample testing folder and the number of testing pairs (clean,noise), in this case there are only 10 images in our github repository folder for different levels of noise; it is possible to configure it to do it for 100, 1000 or the whole test dataset at 57458 testing pairs. The idea is to use it with the [noise_10](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing), [noise_25](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) and [noise_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) testing datasets. <br /> <br />

#### Testing over a single image

In order to test the model over a single image, we will be using [singleImageTester.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/singleImageTester.py) located in [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util) folder.

#### Example of expected output
![singleImageTester_expected_output](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/singleImageTester_expected_output.png)<br /> <br />

####  it is possible to configure the file to change the default path, in this case we have three paths, the noisy image path, the clean image path and the output folder path which defaults to [single_Image_tester_Results](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/single_Image_tester_Results) located at [Util](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util). <br />

####  it is possible to test it with the images from the sample testing folders: [testSample_10](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/testSample_10), [testSample_25](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/testSample_25) or [testSample_50](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/testSample_50). Remember there are only 10 images in our github repository folder for different levels of noise; The idea is to use it with the [noise_10](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing), [noise_25](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) and [noise_50](https://drive.google.com/drive/folders/1PdTrAV-PUpFhdvhFtfOggpLbOpDEouLc?usp=sharing) testing datasets. <br /> <br />


### Creation of noise <br />

#### Following, are the codes which were used for the generation of Additive White Gaussian Noise (AWGN). <br />
#### The noise was created using the numpy library. <br />
#### For fair comparison and reproducibility seeding was employed. <br />

#### The process to create noisy images is displayed below. <br />
#### A clean image is added with a noise map to create a noisy image. <br />
![noise_creation](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/noise_creation.png)
#### Creating noise for the whole [Histopathologic Cancer Detection dataset](https://www.kaggle.com/c/histopathologic-cancer-detection/data)

#### Before you begin <br />

Make sure you have a folder with the images named 1.tif, 2.tif, 3.tif, 4.tif ... etc. it is possible to use [renaming_files_ascending_order](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/ubuntu_snippets/renaming_files_ascending_order) for this task, because the file names from the original dataset are too long. <br />

This is how the image names come by default from the kaggle website <br /><br />
![dataset_long_names](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/dataset_long_names.png)

After running [renaming_files_ascending_order](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/ubuntu_snippets/renaming_files_ascending_order) we will have a folder looking like this <br /><br />
![dataset_short_names](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/dataset_short_names.png)

### If you have the folder that looks like the previous image we can continue <br />
We will be using [multipleImageNoiseCreator.py](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/multipleImageNoiseCreator.py) for corrupting the images between ranges **σ[0,50].** <br />
You need to state the **<<"inputfolder">>**, it does not have a default folder path. <br />

After you have written down the input folder now we need to state the **<<"outputfolder">>** in which the image pairs (clean,noise) will be created.<br />

The expected output folder would look like this <br /> <br />
![averageTester_expected_output](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/README_FILES/multipleImageNoiseCreator_expected_output.png)


#### Creating noise for a single image

Here we will be creating a noisy image using [noiseCreatorSingleImage](https://github.com/Fabio-Gil-Z/IRUNet/blob/main/Code/Util/noiseCreatorSingleImage.py). <br />

We only need to state two things, the path to the clean image and the **outputput folder** which defaults to [noise_Images_Created_By_User](https://github.com/Fabio-Gil-Z/IRUNet/tree/main/Code/Util/noise_Images_Created_By_User) and the expected output can be seen in the same folder. <br />
 it is possible to choose the level of corruption, which is stated as **noise_standard_deviation**. <br /><br /><br /><br /><br />

*That would be it for now, if you have any question / suggestions feel free to send me an email to: fhgil@utp.edu.co, f.gil@studenti.unisa.it* <br />
*Thank you for reading, have a great day!*
