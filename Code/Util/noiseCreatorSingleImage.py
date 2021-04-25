import glob, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.io import imread,imsave

######################################################################################
#                                                                                    #
#                 The following code expects you to use images from                  #
#                histopathologic-cancer-detection datset from kaggle                 #
#          Source https://www.kaggle.com/c/histopathologic-cancer-detection/data     #
#                                                                                    #
######################################################################################


###############################################################
#########      Please state the name of the file      #########
#########  you want to use to corrupt it with noise   #########
###################      EXAMPLES     #########################
###															                            ###
###        cleanImagePath = "testSample_10/1_clean.tif"     ###
###															                            ###
###        cleanImagePath = "testSample_25/2_clean.tif"     ###
###															                            ###
###        cleanImagePath = "testSample_50/3_clean.tif"     ###
###															                            ###
###        cleanImagePath = "testSample_10/4_clean.tif"     ###
###															                            ###
###############################################################
cleanImagePath  = "testSample_10/1_clean.tif"

#Please try to match the names so they are easier to use with the other tools
#For example if you are creating a noisy image with name 55_clean.tif
#The outputFileName should have a matching name such as  55_noise.tif
outputFileName  = "1_noise.tif"


#Please state the name of the outputfolder for the newly created noisy image
#Default name: "noise_Images_Created_By_User/" (This is a folder name, dont get confused)
outputImagePath = "noise_Images_Created_By_User/"



#Loading Image
clean = imread(cleanImagePath)
#Normalizing image
clean = clean / 255.


#For fair comparison and reproducibility we use seed(0)
np.random.seed(0)



###############################################################
#########      Please state the standad deviation     #########
#########               between [0,50]                #########
###############################################################
###################       EXAMPLES       ######################
##############  noise_standard_deviation = 10  ################
##############  noise_standard_deviation = 25  ################
##############  noise_standard_deviation = 50  ################
###############################################################

noise_standard_deviation = 50


#The shape of the input image, leave it as it is
#Becuase the shape of the images from the kaggle dataset are fixed at (96,96,3)
shape           = (96,96,3)



#Generating the noise map with:
#loc   = 0                   ----> mean = 0
#scale = noise_standard_deviation/255 ----> standard deviation and divided by 255 to normalize the map
#size  = shape               ----> dimensions of the image, which are (96,96,3)
noiseMap = np.random.normal(loc=0, scale=noise_standard_deviation/255, size=shape)


#Clipping the image to better approximate real world noise
#According to:  J.-S.  Lee,  “Refined  filtering  of  image  noise  using  local  statistics,”Computer  graphics  and  image  processing,  vol.  15,  no.  4,  pp.  380–389, 1981.
noise  = np.clip((clean + noiseMap),0,1)
noise = noise.astype(np.float32)
noise = noise * 255


#Saving the image to the outputImagePath defined above
#Defaults to: "noise_Images_Created_By_User" (This is a folder name, dont get confused)
imsave(outputImagePath+outputFileName,(noise).astype(np.uint8))
