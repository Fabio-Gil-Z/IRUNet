import time
import tensorflow as tf
import numpy as np
from skimage.io import imread,imsave
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from matplotlib import pyplot



def loadImage(file):
	img = imread(file)
	img = np.expand_dims(img, axis=0)
	img = np.asarray(img)
	img = img.astype('float32') / 255.
	return img

###############################################################
#########      Please state the name of the files     #########
#########        you want to use for testing          #########
###################      EXAMPLES     #########################
###															###
###        cleanImagePath = "testSample_10/1_clean.tif"     ###
###        noiseImagePath = "testSample_10/1_noise.tif"     ###
###															###
###        cleanImagePath = "testSample_25/2_clean.tif"     ###
###        noiseImagePath = "testSample_25/2_noise.tif"     ###
###															###
###        cleanImagePath = "testSample_50/3_clean.tif"     ###
###        noiseImagePath = "testSample_50/3_noise.tif"     ###
###															###
###        cleanImagePath = "testSample_10/4_clean.tif"     ###
###        noiseImagePath = "testSample_10/4_noise.tif"     ###
###															###
###############################################################


cleanImagePath = "testSample_50/4_clean.tif"
noiseImagePath = "testSample_50/4_noise.tif"

results_output_folder = "results_output_folder"


#Please state the folder in which you have the models saved
#Default name: "Models/"
modelPathFolder = "Models/"

#If you trained a network using my code please verify the name of the model
#(The name of the model is the folder name)
#Default name: IRUNet
modelName 	   = "IRUNet"
model     	   = tf.keras.models.load_model(modelPathFolder + modelName)


#Loading image pairs (clean, noise)
clean    = loadImage(cleanImagePath)
noise    = loadImage(noiseImagePath)

#Creating the denoised image with the model
start_time = time.time()
denoised = model.predict(noise)
print("--- %s seconds ---" % (time.time() - start_time))

#Removing the batch dimension from the tensor, from (1,96,96,3) to (96,96,3)
#in order to enable the psnr and ssim calculation
clean    = np.squeeze(clean)
noise    = np.squeeze(noise)	
denoised = np.squeeze(denoised)

#Calculating the psnr between the clean and denoised image
psnr     = peak_signal_noise_ratio(clean, denoised, data_range=None)

#SSIM is calculated on grayscale, so we change the images to gray scale
cleanGray     = rgb2gray(clean)
denoisedGray  = rgb2gray(denoised)
#skimage.io.imread and rgb2gray are used to read in the data and convert it to grayscale.
#In that case, values will be scaled within [0, 1.0]
#for this reason the 'data_range' should be set to 1.0
#Calculating ssim between the clean and denoised image as it was proposed by et al. Wang
#Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13, 600-612.
#https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf, DOI:10.1109/TIP.2003.8198612
ssim = structural_similarity(cleanGray, denoisedGray, multichannel=True, 
gaussian_weights=True, 
sigma=1.5, 
use_sample_covariance=False, 
data_range=1.0)

print("PSNR: {:.6f}    SSIM: {:.6f}".format(psnr,ssim))


plt.figure(figsize=(50,50))
## Noise
ax = plt.subplot(1,3,1)
ax.set_title("Noisy Image", fontsize=20)
plt.imshow(noise)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

## Original
ax = plt.subplot(1,3,2)
ax.set_title("Clean Image", fontsize=20)
plt.imshow(clean)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

## Denoise
ax = plt.subplot(1,3,3)
ax.set_title("Denoised Image", fontsize=20)
plt.imshow(denoised)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.figtext(0.72, 0.22, "PSNR: {:.4f}".format(psnr), ha="center", fontsize=25, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.figtext(0.87, 0.22, "SSIM: {:.4f}".format(ssim), ha="center", fontsize=25, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.show()


imsave(results_output_folder + "/clean.tif",(clean*255).astype(np.uint8))
imsave(results_output_folder + "/noise.tif",(noise*255).astype(np.uint8))
imsave(results_output_folder + "/denoised.tif",(denoised*255).astype(np.uint8))
