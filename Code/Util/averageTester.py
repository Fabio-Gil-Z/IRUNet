import time
import tensorflow as tf
import numpy as np
from skimage.io import imread,imsave
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray



def loadImage(file):
	img = imread(file)
	img = np.expand_dims(img, axis=0)
	img = np.asarray(img)
	img = img.astype('float32') / 255.
	return img

######################################################################################
#                                                                                    #
#Following you may be able to test the PSNR and the SSIM of image pairs from my model#
#                                                                                    #
######################################################################################

#For testing you should have double the amount of image samples, because you need 'image pairs'
#Total number of image pairs on the folder ( e.g. (total folder images / 2))
#For example if you have 100k images, 50k should be clean and 50k should be noisy e.g. (clean,noise)
#Thus the total number of samples would be 50k.

###########################################################################
# Make sure for every clean image, you have its corresponding noise image #
###########################################################################

#In the case of the histopathologic-cancer-detection datset from kaggle
#Source https://www.kaggle.com/c/histopathologic-cancer-detection/data
#The total number of sample test images would be: 57458
#The total number of images on the folder would be: 114,916 (57458 clean, 57458 noise)
#Named as follow:
#1_clean.tif, 1_noise.tif, 2_clean.tif, 2_noise.tif .... etc
#If you do not have the noise images, use my "multipleImageNoiseCreator.py" program in Utils folder
#If you do not have the files named as stated, please use my ubuntu snippets in Utils folder




######################################################################################
#                                                                                    #
#  The following code expects you to have named the files accordingly and are using  #
#  the histopathologic-cancer-detection test datset from kaggle                      #
#                                                                                    #
######################################################################################


#Number of samples you want to obtain the average to.
#For example 100, 1000, 10000 or the whole dataset 57458
#Defaults to: 10 for the images on the github repository
total_number_of_samples = 1000

#Folder path where the image pairs (clean,noise) are stored
#Default names: testSample_10, testSample_25, testSample_50
pathFolder    = "/media/neuronelab/EXP_FABIO1/Downloaded_Datasets/noise_10/"
file_extension  = ".tif"



#Please state the folder in which you have the models saved
#Default name: "Models/"
modelPathFolder = "Models/"

#If you trained a network using my code please verify the name of the model
#(The name of the model is the folder name)
#Default name: IRUNet
modelName       = "IRUNet"


#Loading model
model           = tf.keras.models.load_model(modelPathFolder + modelName)
model.predict(np.zeros((1,96,96,3)))

#Variable to accumulate the psnr acquired of each image it is denoised
accumulated_psnr  = 0
#Variable to accumulate the ssim acquired of each image it is denoised
accumulated_ssim  = 0
#Variable to accumulate the computation time
accumulated_time  = 0
for i in range(1,total_number_of_samples+1):
	#Loading image pairs (clean, noise)
	clean    = loadImage(pathFolder + '/{}_clean'.format(i) + '{}'.format(file_extension))
	noise    = loadImage(pathFolder + '/{}_noise'.format(i) + '{}'.format(file_extension))
	#Measuring computing time
	start_time = time.time()
	#Creating the denoised image with the model
	denoised = model.predict(noise)
	accumulated_time += time.time() - start_time
	#Removing the batch dimension from the tensor, from (1,96,96,3) to (96,96,3)
	#in order to enable the psnr and ssim calculation
	clean    = np.squeeze(clean)
	noise    = np.squeeze(noise)	
	denoised = np.squeeze(denoised)
	#Calculating the psnr between the clean and denoised image
	psnr     = peak_signal_noise_ratio(clean, denoised, data_range=None)
	accumulated_psnr  += psnr	
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
	accumulated_ssim += ssim
	print("Image: {}    PSNR: {:.6f}    SSIM: {:.6f}".format(i,psnr,ssim))
average_psnr = accumulated_psnr / total_number_of_samples
average_ssim = accumulated_ssim / total_number_of_samples
print("Average PSNR: {}".format(average_psnr))
print("Average SSIM: {}".format(average_ssim))

print("Average computation time per image --- %s seconds ---", (accumulated_time/total_number_of_samples))
