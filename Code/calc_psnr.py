import numpy as np
import math
import random
from skimage.metrics import peak_signal_noise_ratio

def psrn_on_callBack(x_noise,x_target,autoencoder,randomImage):
    noiseImage = x_noise[randomImage]
    cleanImage = x_target[randomImage]
    noiseImage = np.expand_dims(noiseImage, axis=0)
    denoised   = autoencoder.predict(noiseImage)
    denoised   = np.squeeze(denoised)
    cleanImage = np.squeeze(cleanImage)
    noiseImage = np.squeeze(noiseImage)
    psnr       = peak_signal_noise_ratio(denoised, cleanImage, data_range=None)
    return psnr, denoised, cleanImage, noiseImage


