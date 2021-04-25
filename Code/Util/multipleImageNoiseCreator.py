import glob, os, cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.io import imread,imsave

######################################################################################
#                                                                                    #
#                     The following code expects you to use the                      #
#              histopathologic-cancer-detection TRAINING datset from kaggle          #
#        Source https://www.kaggle.com/c/histopathologic-cancer-detection/data       #
#                                                                                    #
######################################################################################


######################################################################################
#                                                                                    #
#				BEFORE YOU BEGIN                                     #
#                                                                                    #
######################################################################################

######################################################################################
#                                                                                    #
#		            IMPORTANT PRE PROCESSING STEP                            #
#                                                                                    #
######################################################################################


######################################################################################
#                                                                                    #
#                                  RENAMING FILES                                    #
#                                                                                    #
######################################################################################
#      Please make sure that the file names are named in asceding order              #
#      For example  1.tif, 2.tif, 3.tif, 4.tif, 5.tif, 6.tif, and so on              #
#      If you do not have the files named this way, please use my                    #
#      renaming_"files_ascending_order" snippet on the 'Utils' folder                #
######################################################################################



######################################################################################
#                                                                                    #
#			           !!!WARNING!!                                      #
#                                                                                    #
######################################################################################
#            The following code will create a training dataset with                  #
#            image pairs (clean,noise), it will be stored in your                    #
#            selected "outputfolder" below.                                          #
#            If you are going to use the TRAINING kaggle dataset                     #
#            It will create a total of 440050 images                                 #
#            220025 clean images and 220025 noise images                             #
#            Leaving your initial training folder untouched                          #
######################################################################################
#                                                                                    #
#                                      REMINDER                                      #
#                                                                                    #
######################################################################################
#            The following code expects that the image files are named               #
#            in asceding order                                                       #
#            For example  1.tif, 2.tif, 3.tif, 4.tif, 5.tif, 6.tif, and so on        #
#            If you do not have the files named this way, please use my              #
#            renaming_"files_ascending_order" snippet on the 'Utils' folder          #
######################################################################################






#Please state the folder path for the TRAINING dataset from kaggle
#Default name: << does not have default folder name >> USER MUST SPECIFY IT.

inputfolder   =  ""


#The shape of the input image, leave it as it is
#Because the shape of the images from the kaggle dataset are fixed at (96,96,3)
shape         = (96,96,3)



#List in which we are going to save the noise maps created
maps_of_noise = []


#List of possible standard deviation corruption, goes from 0 to 50.
noise_standard_deviation_list = np.arange(51)


#For each standard deviation, we are going to create a noise map
#and store it in our list of noise maps variable "maps_of_noise"
for noise_standard_deviation in noise_standard_deviation_list:
	#For fair comparison and reproducibility we use seed(0)
	np.random.seed(0)
	#Generating the noise map with:
	#loc   = 0                            ----> mean = 0
	#scale = noise_standard_deviation/255 ----> standard deviation and divided by 255 to normalize the map
	#size  = shape                        ----> dimensions of the image, which are (96,96,3)
	maps_of_noise.append(np.random.normal(loc=0, scale=noise_standard_deviation/255, size=shape))


#Inside the for loop below we will be increasing the standard deviation
#In order to create maps of 0 to 50 standard deviation
#Once the standard deviation hits 50 we reset it to 0
#This way we will be creating 4400 images of each standard deviation
#In other words: 
#4400 for standar deviation 1
#4400 for standar deviation 2
#4400 for standar deviation 3
#...
#...
#...
#4400 for standar deviation 49
#4400 for standar deviation 50
#4400 for standar deviation 1
#4400 for standar deviation 2
#4400 for standar deviation 3
#And repeat until we reach 220025 images
standard_deviation_selector = 0


#Outputfolder for the training dataset that will be created
#Remember we need image pairs (clean,noise) to train the network
#Defaults to: "user_self_created_dataset/" folder (This is a folder name, dont get confused)
outputfolder    = "user_self_created_dataset/"
#The TRAINING dataset from kaggle has a total of 220025 images
#This is where the number in the for loop is from
#Feel free to modify it in case you are using less images
for i in range(1,220025):
	if standard_deviation_selector == 51:
		standard_deviation_selector = 0
	img   = imread(inputfolder + "{}.tif".format(i))
	img   = img / 255.
	#Clipping the image to better approximate real world noise
	#According to:  J.-S.  Lee,  “Refined  filtering  of  image  noise  using  local  statistics,”Computer  graphics  and  image  processing,  vol.  15,  no.  4,  pp.  380–389, 1981.
	noise = np.clip((img + maps_of_noise[standard_deviation_selector]),0,1)
	noise = noise.astype(np.float32)
	noise = noise * 255
	img   = img * 255
	imsave(outputfolder + "{}_noise.tif".format(i),noise.astype(np.uint8))
	imsave(outputfolder + "{}_clean.tif".format(i),img.astype(np.uint8))
	print("Current standard_deviation_selector: ",standard_deviation_selector)
	standard_deviation_selector += 1


######################################################################################
#                                                                                    #
#					 Your freshly created dataset has been created                   #
#                          and ready to be used for training                         #
#                                                                                    #
######################################################################################
