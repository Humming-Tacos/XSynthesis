import asyncio

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from .common import bcolors


PROC_NAME = f'{bcolors.BOLD}[COMPRESSOR]{bcolors.ENDC}'


def level_detect(x, fs: int, attack_ms: float = 0.1, release_ms: float = 100):
    """ Detects levels of a given audio.


    Args:
        x (wav_file): the contents of a input wav file.
        fs (int): the sample rate of the input wav file.
        attack_ms (float, optional): The attack duration. Defaults to 0.1.
        release_ms (int, optional): The release duration. Defaults to 100.

    Returns:
        y: An array of detected levels corresponding to each sample of the wav file.
    """
    # normalize input samples according to overall maxima
    x_normalized = x / np.max(x)
    print(PROC_NAME, 'Normalized input')
 
    # obtain the maximum vector from both channels
    print(PROC_NAME, 'Calculating maxes')
    x_maxes = np.max(np.abs(x_normalized), axis=1)
    # x_maxes = [max(np.abs(x_t)) for x_t in tqdm(x_normalized)]
    print(PROC_NAME, 'Calculated maxes')


    def _calc_coefs(tau_ms, fs):
        a1 = np.exp(-1.0 / (fs * tau_ms / 1000.0))
        b0 = 1.0 - a1
        return b0
    b0_a = _calc_coefs(attack_ms, fs)
    b0_r = _calc_coefs(release_ms, fs)

    y = np.copy(x_maxes)
    level_est = 0

    for n, x_samp in tqdm(enumerate(x_maxes)):
        if x_samp > level_est:  # attack mode
            level_est += b0_a * (x_samp - level_est)
        else:  # release mode
            level_est += b0_r * (x_samp - level_est)
        y[n] = level_est

   
    return y


async def compress_async(input_path: str, output_path: str, ratio: float, write: bool, parallel: bool):
    """Async helper function to compresses audio with given ratio and threshold.


    Args:
        input_path (path): Path to the input wav file.
        output_path (path): Path to the output wav file.
        ratio (float): Ratio of compression.
        write (bool): Flag to output compressed wav as a file.
        parallel (bool): Flag to enable multithread compressing.
    """

    def _calculate_threshold(levels, ratio):
        integrated_loudness = np.sum(levels) / levels.size
        maximum_loudness = np.max(levels)
        loudness_range = np.sqrt(maximum_loudness - integrated_loudness)
        magic_number_constant = 2 # :D
        threshold = 1 - (loudness_range - integrated_loudness) ** ((ratio-1)*magic_number_constant) 
        print(PROC_NAME, 'Detected threshold:', threshold)
        return threshold

    def _compress_sample(x_l, x_r, level, ratio, threshold):
        if level < threshold:
            return [x_l, x_r]
        overflow = level - threshold
        return [x_l * (1 - overflow/ratio) , x_r * (1 - overflow/ratio) ] 

    def _compress_slice(x_slice, level_slice, ratio, threshold, t):
        print(PROC_NAME, f'EXECUTING {t}')
        compressed_slice = []
        for x_t, level_t in zip(x_slice, level_slice):
            compressed_slice.append(_compress_sample(*x_t, level_t, ratio, threshold))
        print(PROC_NAME, f'!DONE {t}')
        return np.asarray(compressed_slice)


    # read file input
    fs, x = wavfile.read(input_path)
    print(PROC_NAME, f'Reading from path: {input_path}')

    # detect levels of the maxium vector
    print(PROC_NAME, 'Detecting levels')
    levels = level_detect(x, fs)
    print(PROC_NAME, 'Levels detected')

    print(PROC_NAME, 'Calculating Threshold')
    threshold = _calculate_threshold(levels, ratio)

    
    # apply compression on both channels
    
    print(PROC_NAME, 'Creating Compressor Tasks')
    
    if not parallel:
        progress_bar = tqdm(zip(x, levels))
        data_out = np.asarray([_compress_sample(*x_t, level_t, ratio, threshold) for x_t, level_t in progress_bar])
    else:
        loop = asyncio.get_event_loop()

        tasks = []

        total_sample = levels.size
        slice_length = 5*fs

        for t in tqdm(range(0, total_sample, slice_length)):
            tasks.append(loop.run_in_executor(None, _compress_slice, x[t:t+slice_length], levels[t:t+slice_length], ratio, threshold, t/slice_length))

        print(PROC_NAME, 'Executing Compressor Tasks in parallel')
        data_out = await asyncio.gather(*tasks)
        print(PROC_NAME, 'Restacking results')
        data_out = np.vstack(data_out)
        print(data_out.shape)
 
    # export audio
    print(PROC_NAME, f'Writting to path: {output_path}')
    if write:
        wavfile.write(output_path, fs, data_out.astype(x.dtype))
    

def compress_audio(input_path: str, output_path: str, ratio: float = 2, write: bool = True, parallel: bool = False):
    """Compresses audio with given ratio and threshold.

    Args:
        input_path (path): Path to the input wav file.
        output_path (path): Path to the output wav file.
        ratio (float): Ratio of compression. Defaults to 2.
        write (bool): Flag to output compressed wav as a file. Defaults to True.
        parallel (bool): Flag to enable multithread compressing. Defaults to False.
    """    

    asyncio.run(compress_async(input_path, output_path, ratio, write, parallel))


def compress(wav_path: str , verbose: bool = False) -> str:
    """XSynthesis' compress function. Compresses given audio wav.

    Args:
        wav_path (path): Path to the input wav file.
        verbose (bool, optional): Flag to enable verbose plots and logs. Defaults to False.

    Returns:
        wav_path_post_compress (path): Path to the compressed output wav file.
    """    
    wav_path_post_compress = ''.join(wav_path.split('.')[:-1])+'_compressed.wav'
    compress_audio(wav_path, wav_path_post_compress, ratio=1.5, write=True)
    return wav_path_post_compress
    
