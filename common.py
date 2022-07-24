import numpy as np
import scipy.io.wavfile as wav

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# utility functions
def normalize_by_integration(arr):
    arr = np.asarray(arr)
    integration = np.sum(arr)
    return arr / integration


def db2lin(db):
    return np.power(10, db/20)


def lin2db(lin):
    return 20 * np.log10(lin)


def generate_whitenoise(seconds, fs):
    """
    Example
    ----------
    fs = 44100

    seconds = 60
    
    generate_whitenoise(seconds, fs)
    """
    white_noise_vector = np.random.rand(seconds*fs)*(2**15) - (2**14)
    white_noise_sound = np.stack((white_noise_vector, white_noise_vector), axis=1)
    wav_name = f'whitenoise_{fs}_{seconds}.wav'
    wav.write(wav_name, fs, white_noise_sound)
    return wav_name
    