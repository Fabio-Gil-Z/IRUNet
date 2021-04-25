import os,random,cv2,time,glob
import time,math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPool2D,BatchNormalization,ReLU,Add,Concatenate
import numpy as np

import load_dataset
import IRUNet
import calc_psnr

from skimage.io import imread,imsave


##############################################################
#                          MODEL CONFIG                      #
##############################################################


###############################
#       LOADING DATASETS      #
###############################


#Please state your batch size (It is needed for the generator)
#The generator is used to fetch the images from the file name
#Because loading all the images on memory is not possible
BATCH_SIZE = 32 


#Please state the directoy path in which the dataset named 'noise_0_to_50' is stored
#Or if you are using your own dataset, make sure there are pair of images named as follows
#1_clean.tif, 1_noise.tif, 2_clean.tif, 2_noise.tif, etc
#If you do not have the noise images, please use my noise generator program 'multipleImageNoiseCreator.py' in Utils folder
#If you do not have the files named as stated, please use my ubuntu snippets in Utils folder
DATASET_DIRECTORY = "/media/neuronelab/EXP_FABIO1/Downloaded_Datasets/temp/"


#Please state the directory in which you want to save the model
#Default name: Models
DIRECTORY_TO_SAVE_MODELS = "Models"

#Please state the directory in which you want to save the model
#Default name: Tensorboard
DIRECTORY_TO_SAVE_TENSORBOARD = "Tensorboard"


#Please state the directory in which images will saved
#Default name: Epoch_results
#The spected output should be: clean.tif , noise.tif
#And denoised_epoch_psnr.tif
#For example a file named "denoise_4_24.55.tif" can be interpreted as:
#The first '4' denotes the epoch, this is an image obtained at epoch 4
#The number '24.55' denotes the psrn obtaned after denoising it
DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT = "Epoch_results"

#Loads the file names of the images and stores them in an arrays
#for clean images and noise images
x_noise,x_clean = load_dataset.load_dataSet(DATASET_DIRECTORY)

#Uses the previous arrays to load images equal to the batch size selected ( in this case 32)
generator       = load_dataset.testSequence(x_noise,x_clean,BATCH_SIZE)
###############################
#    END LOADING DATASETS     #
###############################





#Please state the name you want to save the model with
modelName = "myDenoiser"

#Please state the name of the model you want to load the weights from
#This can be used to resume training in case it had to be stopped
#Default name: same as <modelName>
#For example if your model is named: myModel_2
#Then you should put myModel_2 in the variable "weights" too
weights = "myDenoiser"


#Used to load the weights, (leave it, as it is)
checkpointName = "checkpoint_{}".format(weights)


#The "loadWeights" variable is used to resume your training in case it stopped
#True for loading weights and resume training
#False for a fresh start in training a network (default state 'False')
loadWeights = False


#Please state the number of epochs you want for train the network
epoch_assignation = 1000

## Image Size
H = 96
W = 96
C = 3


#Different optimizer to be used in training
SGD_optimizer = tf.keras.optimizers.SGD(
    learning_rate=1e-4, momentum=0.9, nesterov=False, name='SGD')
ADAM_optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
    name='Adam')
ADADELTA_optimizer = tf.keras.optimizers.Adadelta(
    learning_rate=0, rho=0.95, epsilon=1e-7, name='Adadelta'
)


#Please state the number of filters you want to use in the network
filters         = 16


#Please state an optimizer to use, preferably choose one of the above
optimizer       = ADAM_optimizer

#Please state a loss function to use (default: MeanAbsoluteError)
#You may try: "MeanSquaredError"
loss_function   = "MeanAbsoluteError"
##############################################################
#                      END MODEL CONFIG                      #
##############################################################



#Loading the model from IRUNet.py file which was imported at the beginning
model = IRUNet.IRUNet(H,W,C,filters)

model.summary()


model.compile(optimizer=optimizer, loss=loss_function)

class CustomModelCheckPoint(tf.keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.epoch_accuracy = {} # loss at given epoch
        self.epoch_loss = {} # accuracy at given epoch
        self.epoch_of_best_psrn  = 1
        self.previous_psnr       = 0
        self.printEpoch          = 50

    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch.
        return

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        self.epoch_accuracy[epoch] = logs.get("acc")
        self.epoch_loss[epoch] = logs.get("loss")
        ####################TEST THE MODEL WITH A GIVEN IMAGE###########################
        #item = random.randint(0,31)
        item = 0
        gen_noise     = generator.__getitem__(item)[0]
        gen_target    = generator.__getitem__(item)[1]
        PSNR, denoised,cleanImage, noiseImage = calc_psnr.psrn_on_callBack(gen_noise,gen_target,model,item)
        print("\n\nPSNR:{}".format(PSNR))
        
        # SAVING THE MODEL AT THE END OF EVERY EPOCH IF THE PSNR IS LOWER
        if PSNR > self.previous_psnr:
            self.epoch_of_best_psrn = epoch+1
            self.previous_psnr = PSNR
            model.save(DIRECTORY_TO_SAVE_MODELS + "/{}".format(modelName))
        denoised = denoised * 255
        imsave(DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT + "/denoised_{}_{:.2f}.tif".format(epoch+1,PSNR),denoised.astype(np.uint8))

        
        if epoch+1 == 1:
            cleanImage = cleanImage * 255
            imsave(DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT + "/clean.tif",cleanImage.astype(np.uint8))

            noiseImage = noiseImage * 255
            imsave(DIRECTORY_TO_SAVE_EACH_EPOCH_RESULT + "/noise.tif",noiseImage.astype(np.uint8))
        

        print("Best PSNR: ", self.previous_psnr)
        print("Epoch of best PSRN: ",self.epoch_of_best_psrn)
        print("ok")



checkpoint = CustomModelCheckPoint()
# SAVING THE WEIGHTS TO CONTINUE TRAINING LATER
checkpoint_filepath = DIRECTORY_TO_SAVE_MODELS + "/{}/{}".format(checkpointName,weights)

training_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

tensorBoard = tf.keras.callbacks.TensorBoard(
    log_dir = DIRECTORY_TO_SAVE_TENSORBOARD + "/{}".format(modelName),
    histogram_freq=1,
    embeddings_freq=1
    )

#tensorboard --logdir='/media/neuronelab/EXP_FABIO1/IRUNet/Tensorboard/'
#http://localhost:6006


callbacks_list = [checkpoint,training_checkpoint,tensorBoard]


if loadWeights == True:
    print("LOADING WEIGHTS = TRUE")
    print("Weights loaded from model: '{}'".format(weights))
    model.load_weights(DIRECTORY_TO_SAVE_MODELS + "/{}/{}".format(checkpointName,weights))
else:
    print("LOADING WEIGHTS = FALSE")



model.fit(
    generator,
    epochs=epoch_assignation,
    steps_per_epoch=None,
    shuffle=True,
    workers=5,
    callbacks=callbacks_list,
)



