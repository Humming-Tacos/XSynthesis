import asyncio
import json
import os
from itertools import chain

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from .common import bcolors, db2lin, normalize_by_integration

PROC_NAME = f'{bcolors.BOLD}[EQUALIZER]{bcolors.ENDC}'

# NOTE: The default path to the eq curve numpy file, change in production
default_target_eq_curve = 'XSynthesis/configurations/average_responses_Piano Pop Song TRAINING.npy'


def calculate_avg_frequency_response(wav_path, hop_length=1024, win_length=2048, ma_width=32, smooth=False):
    """Step 1 of eq: Calculate average frequency response of an input WAV file.

    Args:
        wav_path (path): Path to the input wav file.
        hop_length (int, optional): Hop length of each fft frame. Defaults to 1024.
        win_length (int, optional): Window length of each fft frame. Defaults to 2048.
        ma_width (int, optional): Moving average's window width when smoothing is enabled. Defaults to 32.
        smooth (bool, optional): Flag to enable smoothing. Defaults to False.

    Returns:
        avg_freq_response (list): The average frequency response of the given wav audio.
    """
    wav_data, _ = librosa.load(wav_path)
    stft_data = np.abs(librosa.stft(wav_data, hop_length=hop_length, win_length=win_length))

    cut_off_bin = 32
    avg_freq_response = np.asarray([ sum(window) for window in stft_data ]) 

    avg_freq_response = normalize_by_integration(avg_freq_response)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if smooth:
        ma_avg_freq_response = np.append(
            avg_freq_response[:ma_width-1],
            moving_average(avg_freq_response, ma_width)
        )

    if smooth:
        avg_freq_response[cut_off_bin:] = ma_avg_freq_response[cut_off_bin:]

    return avg_freq_response


def target_error_from_json(existing_curve, target_curve_json_path, verbose=False):
    """Step 2 of eq: Calculate difference between target curve and our existing curve. (Using Json)

    Args:
        existing_curve (list): The existing average response of an input audio.
        target_curve_json_path (path): The json path to the target response curve. 
        verbose (bool, optional): Flag to enable verbose. Defaults to False.

    Returns:
        error (list): The error between the existing average response and the target curve.
    """
    def clean_json(curve_dict, dict_name, un_log=True):
        curve = [entry['Value'] for i, entry in  enumerate(curve_dict[dict_name]['Value']) if i % 2 == 0]
        if un_log:
            curve = normalize_by_integration([db2lin(db) for db in curve])
        return curve

    with open(target_curve_json_path, 'r') as target_curve_json_file_ptr:
        target_curve_dict = json.load(target_curve_json_file_ptr)
        normalized_mag_lin = clean_json(target_curve_dict, 'normalized_mag_dB')

        error = [((target / existing)+1)/2 for existing, target in zip(existing_curve, normalized_mag_lin)]

        if verbose:
            x_axis = np.arange(0, 22050, 22050 / 1025)
            plt.plot(x_axis, normalized_mag_lin, label=' '.join(target_curve_json_path.split('.')[:-1]) + ' Target Curve')
            plt.plot(x_axis, existing_curve, label='existing_curve')
            plt.legend()

            plt.title('Average frequency response vs. Target curve')
            plt.ylabel('Absolute Amplitude')
            plt.xlabel('Frequency [Hz]')

            plt.xscale('log')
            plt.yscale('log')
            plt.show()

        return error


def target_error_from_numpy(existing_curve, target_curve_numpy_path, verbose=False):
    """Step 2 of eq: Calculate difference between target curve and our existing curve. (Using numpy)

    Args:
        existing_curve (list): The existing average response of an input audio.
        target_curve_numpy_path (path): The numpy path to the target response curve. 
        verbose (bool, optional): Flag to enable verbose. Defaults to False.

    Returns:
        error (list): The error between the existing average response and the target curve.
    """
    target_curve = np.load(target_curve_numpy_path)
    
    error = [((target / existing)+1)/2 for existing, target in zip(existing_curve, target_curve)]
    # error = [target / existing for existing, target in zip(existing_curve, target_curve)]
    if verbose:
        x_axis = np.arange(0, 22050, 22050 / 1025)
        plt.plot(x_axis, target_curve, label=f'{target_curve_numpy_path} Target Curve')
        plt.plot(x_axis, existing_curve, label='Existing Curve')
        plt.legend()

        plt.title('Average frequency response vs. Target curve')
        plt.ylabel('Absolute Amplitude')
        plt.xlabel('Frequency [Hz]')

        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    if verbose:
        x_axis = np.arange(0, 22050, 22050 / 1025)
        plt.plot(x_axis, error, label='Error curve')

        plt.title('Error curve')
        plt.ylabel('Linear Error')
        plt.xlabel('Frequency [Hz]')

        plt.xscale('log')

        plt.show()

    return error 


async def process_channel(data, error, fs, verbose=False):
    """EQ filter for one channel of input audio asyncrhonously.

    Args:
        data (list): Input audio samples.
        error (list): Difference between the input audio and the target response curve.
        fs (int): Sample rate of the input audio
        verbose (bool, optional): Flag to enable verbose. Defaults to False.

    Returns:
        bands_sum (list): the audio samples post eq.
    """
    def search_error(arr, key, quanta= 22050 / 1025):
        if verbose:
            print(PROC_NAME, key, int(key // quanta), arr[int(key // quanta)])
        return arr[int(key // quanta)]
        
    hardcode_bands = [
        15, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 
        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 
        12500, 16000, 20000, 24100]

    band_count = 30
    
    def average_pair(arr):
        pairs = []
        for i in range(1,len(arr)-1):
            fc1 = np.mean(arr[i-1:i+1])
            fc2 = np.mean(arr[i:i+2])
            pairs.append((fc1, fc2-1))
        return pairs

    fcs = average_pair(hardcode_bands)

    def calculate_fc(band):
        return fcs[band-1]

    def project_mean(fc1, fc2):
        return np.power(10, np.mean([np.log10(fc1), np.log10(fc2)]))

    gains = [search_error(error, project_mean(*fc)) for fc in fcs]

    def calculate_gain(band):
        return gains[band-1]

    def process_band(band, fc, gain, fs, data):
        PROC_NAME = f'[PROCESS-BAND:{band}]'
        def _bandpass_filter(data, lowcut, highcut, fs, gain, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='bandpass')
            filtered = filtfilt(b, a, data)
            return filtered * gain 
        order = 2
        filtered_band = _bandpass_filter(data, *fc, fs, gain=gain, order=order)
        
        print(PROC_NAME, f'bandpass_filter(fc=[{fc[0]:.2f},{fc[1]:.2f}], gain={gain:.2f}, order={order})')
        return filtered_band


    # initialize loop object
    loop = asyncio.get_event_loop()

    tasks = []
    for band in tqdm(range(1, band_count+1)):
        tasks.append(loop.run_in_executor(None, process_band, band, calculate_fc(band), calculate_gain(band), fs, data))
    

    print(PROC_NAME, 'Created all tasks')
    bands = await asyncio.gather(*tasks)
    print(PROC_NAME, 'Processed all tasks')
    print(PROC_NAME, 'Summing bands')
    # bands_sum = np.sum(bands, axis=0)
    
    bands_sum = None
    for band in tqdm(bands):
        if bands_sum is None:
            bands_sum = band
        else:
            bands_sum += band

    print(PROC_NAME, 'Summed bands')
    return bands_sum


def apply_eq(input_wav_path, output_wav_path, error, verbose=False, mirror=False):
    """Step 3 of eq: Apply EQ to a WAV file.

    Args:
        input_wav_path (path): The path to the input wav file.
        output_wav_path (path): The path to the output wav file.
        error (list): Difference between the input wav audio and the target curve.
        verbose (bool, optional): Flag to enable verbose. Defaults to False.
        mirror (bool, optional): Only processes the left channel, then copy results of the left channel to the right. Defaults to False.
    """
    print(PROC_NAME, f'Reading from path: {input_wav_path}')
    freq_s, data_in = wav.read(input_wav_path)
    data = data_in / 4

    equalized = data.copy()

    print(PROC_NAME, 'Processing left:')
    equalized[:,0] = asyncio.run(process_channel(data[:,0], error, freq_s, verbose=verbose))

    if not mirror:
        print(PROC_NAME, 'Processing right:')
        equalized[:,1] = asyncio.run(process_channel(data[:,1], error, freq_s, verbose=verbose))
    else:
        print('[EQUALIZER]', 'Copying left to right:')
        equalized[:,1] = equalized[:,0] 

    equalized = np.nan_to_num(equalized)
    

    print(PROC_NAME, f'Writing to path: {output_wav_path}')
    wav.write(output_wav_path, freq_s, equalized.astype(data_in.dtype))


# TODO: This is a dummy stream function, 之后要和前端组对接
def stream(audio_segment, offset, offset_end, dtype, output_wav_path, fs=44100):
    """Streams audio segment to the front end

    Args:
        audio_segment (list): Segment of audio samples.
        offset (int): Offset to of the segment.
        offset_end (int): Offset plus the segment sample size.
        dtype (type): The value type of the audio sample.
        fs (int, optional): The sample rate of the given audio segment. Defaults to 44100.

    Returns:
        generator: a generator that yields the content of the output segment ts file.
    """    
    PROC_NAME = f'[Stream {offset}:{offset_end}]'

    print(PROC_NAME, audio_segment[:2])

    output_directory = '/'.join(output_wav_path.split('/')[:-1])
    output_name = ''.join(output_wav_path.split('/')[-1].split('.')[:-1])

    segment_prefix = f'{output_directory}/segments/{output_name}_seg_{offset}_{offset_end}'
    segment_wav_path = segment_prefix + '.wav'
    segment_mp3_path = segment_prefix + '.mp3'
    segment_ts_path = segment_prefix + '.ts'

    # save audio segment as .wav
    wav.write(segment_wav_path, fs, audio_segment.astype(dtype))

    # convert audio segment to .ts
    wav_to_mp4 = f'ffmpeg -i {segment_wav_path} {segment_mp3_path}'
    mp4_to_ts = f'ffmpeg -i  {segment_mp3_path} -vcodec copy -acodec copy -vbsf h264_mp4toannexb {segment_ts_path}'
    os.system(wav_to_mp4)
    os.system(mp4_to_ts)

    # wav.write()
    def _generator():
        with open(segment_ts_path, 'rb') as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)

    return _generator()


def stream_eq(input_wav_path, output_wav_path, error, verbose=False):
    """Step 3 of eq: Apply EQ to a WAV file, and stream while processing it.

    Args:
        input_wav_path (path): The path to the input wav file.
        output_wav_path (path): The path to the output wav file.
        error (list): Difference between the input wav audio and the target curve.
        verbose (bool, optional): Flag to enable verbose. Defaults to False.

    Returns:
        stream_generators (generator): generator that yields the audio contents post eq.
    """
    print(PROC_NAME, f'Reading from path: {input_wav_path}')
    freq_s, data_in = wav.read(input_wav_path)
    data = data_in / 4

    fifteen_seconds = 15 * freq_s

    equalized = data.copy()
    block_offset_backup = None

    stream_generators = None

    # if only one block exist
    if data[:,0].size < fifteen_seconds:
        print(PROC_NAME, f'Processing left:')
        equalized[:,0] = asyncio.run(process_channel(data[:,0], error, freq_s, verbose=verbose))
        print(PROC_NAME, f'Processing right:')
        equalized[:,1] = asyncio.run(process_channel(data[:,1], error, freq_s, verbose=verbose))
        stream_generators = stream(equalized , 0, data[:,0].size, dtype=data_in.dtype, output_wav_path=output_wav_path, fs=freq_s)
   
    # otherwise:
    else:
        for block_num, block_offset in tqdm(enumerate(range(0, data[:,0].size, fifteen_seconds))):
            block = data[block_offset : block_offset+fifteen_seconds]
            
            if len(block) < fifteen_seconds:
                block_offset_backup = block_offset
                block = data[-fifteen_seconds:]
                block_offset = data[:,0].size - fifteen_seconds

            print(PROC_NAME, f'Processing left:{block_num}')
            equalized[block_offset : block_offset+fifteen_seconds,0] = asyncio.run(process_channel(block[:,0], error, freq_s, verbose=verbose))
            print(PROC_NAME, f'Processing right:{block_num}')
            equalized[block_offset : block_offset+fifteen_seconds,1] = asyncio.run(process_channel(block[:,1], error, freq_s, verbose=verbose))

            # get stream generator
            if block_offset_backup is None:
                stream_generator = stream(equalized[block_offset : block_offset+fifteen_seconds] , block_offset, block_offset+fifteen_seconds, dtype=data_in.dtype, output_wav_path=output_wav_path, fs=freq_s)
            else:
                stream_generator = stream(equalized[block_offset_backup : ] , block_offset_backup, data[:,0].size, dtype=data_in.dtype, output_wav_path=output_wav_path, fs=freq_s)
            
            # chain generators
            if stream_generators is None:
                stream_generators = stream_generator
            else:
                stream_generators = chain(stream_generators, stream_generator)
        
    equalized = np.nan_to_num(equalized)

    print(PROC_NAME, f'Writing to path: {output_wav_path}')
    wav.write(output_wav_path, freq_s, equalized.astype(data_in.dtype))

    return stream_generators


# interfaces
def auto_eq_v2(input_wav_path, output_wav_path, target_eq_raw, verbose=False, write=True, isJson=False, stream=True):
    """Automatically calculates and and apply EQ to an WAV audio given a target EQ curve.

    Args:
        input_wav_path (path): Path to the input wav file.
        output_wav_path (path): Path to the output wav file post EQ.
        target_eq_raw (path): Path to the target EQ curve.
        verbose (bool, optional): Flag to enable verbose. Defaults to False.
        write (bool, optional): Flag to output post eq wav as a file. Defaults to True.
        isJson (bool, optional): Flag indicating if target eq is a Json file. Defaults to False.
        stream (bool, optional): Flag to enable streaming. Defaults to True.
    """
    avg_frequency_response = calculate_avg_frequency_response(input_wav_path)
    target_error = target_error_from_json if isJson else target_error_from_numpy
    error_target = target_error(avg_frequency_response, target_eq_raw, verbose=verbose)
    if write:
        if not stream:
            apply_eq(input_wav_path, output_wav_path, error_target, verbose=verbose)
        else:
            # create response with this generator?
            stream_generator = stream_eq(input_wav_path, output_wav_path, error_target, verbose=verbose)
            
        
def test_auto_eq_v2(input_wav_path, output_wav_path, target_eq_raw, isJson=False):
    """Debug test eq function

    Args:
        input_wav_path (path): Path to the input wav file.
        output_wav_path (path): Path to the output wav file post EQ.
        target_eq_raw (path): Path to the target EQ curve.
        isJson (bool, optional): Flag indicating if target eq is a Json file. Defaults to False.
    """    
    def compare_curve_json(existing_curve_input, existing_curve_output, target_curve_json_path):
        def clean_json(curve_dict, dict_name, un_log=True):
            curve = [entry['Value'] for i, entry in  enumerate(curve_dict[dict_name]['Value']) if i % 2 == 0]
            if un_log:
                curve = normalize_by_integration([db2lin(db) for db in curve])
            return curve

        with open(target_curve_json_path, 'r') as target_curve_json_file_ptr:
            target_curve_dict = json.load(target_curve_json_file_ptr)
            normalized_mag_lin = clean_json(target_curve_dict, 'normalized_mag_dB')
            x_axis = np.arange(0, 22050, 22050 / 1025)
            plt.plot(x_axis, normalized_mag_lin, label=' '.join(target_curve_json_path.split('.')[:-1]) + ' Target Curve')
            plt.plot(x_axis, existing_curve_input, label='Before EQ')
            plt.plot(x_axis, existing_curve_output, label='After EQ')
        
            plt.legend()
            plt.title('Comparison')
            plt.ylabel('Absolute Amplitude')
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
    
    def compare_curve_numpy(existing_curve_input, existing_curve_output, target_curve_numpy_path):
        target_curve = np.load(target_curve_numpy_path)
        x_axis = np.arange(0, 22050, 22050 / 1025)
        plt.plot(x_axis, target_curve, label=f'{target_curve_numpy_path} Target Curve')
        plt.plot(x_axis, existing_curve_input, label='Before EQ')
        plt.plot(x_axis, existing_curve_output, label='After EQ')

        plt.legend()

        plt.title('Average frequency response vs. Target curve')
        plt.ylabel('Absolute Amplitude')
        plt.xlabel('Frequency [Hz]')

        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        

    def compare_fft(input_wav_path, output_wav_path):
        fs, data_in = wav.read(input_wav_path)
        fs, data_out = wav.read(output_wav_path)

        data_in,data_out= data_in[:fs*10],data_out[:fs*10]

        N = len(data_in)
        t  = 1/fs * np.arange(N) 
        f  = fs/N * np.arange(N)

        #computing fft of original signal
        fft_in = np.fft.fft(data_in[:,0])/N

        #computing fft of filtered signal
        fft_out = np.fft.fft(data_out[:,0])/N

        plt.figure(figsize=(10, 8))
        plt.subplot(2,2,1)
        plt.plot(t, data_in[:,0],'-r',label=r"$Original amplitude(t)$")
        plt.xlabel('time[s]')
        plt.legend()
        plt.grid()
        plt.subplot(2,2,2)
        plt.plot(t, data_out[:,0],'-b',label=r"$Filtered amplitude(t)$")
        plt.xlabel('time[s]')
        plt.legend()
        plt.grid()

        plt.subplot(2,2,3)
        plt.plot(f[:N//2],np.abs(fft_in[:N//2]),'-r',label=r"$Original magnitude(f)$")
        plt.xlabel('f [Hz]')
        plt.xlim([0,5e3])
        plt.legend()
        plt.grid()


        plt.subplot(2,2,4)
        plt.plot(f[:N//2],np.abs(fft_out[:N//2]),'-b',label=r"$Filtered magnitude(f)$")
        plt.xlabel('f [Hz]')
        plt.xlim([0,5e3])
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
   
    compare_curve = compare_curve_json if isJson else compare_curve_numpy

    avg_frequency_response_input = calculate_avg_frequency_response(input_wav_path)
    avg_frequency_response_output = calculate_avg_frequency_response(output_wav_path)
    compare_curve(avg_frequency_response_input, avg_frequency_response_output, target_eq_raw)
    compare_fft(input_wav_path, output_wav_path)


def eq(wav_path, target_eq_raw=default_target_eq_curve, isJson=False, verbose=False):
    """XSynthesis' EQ function.

    Args:
        wav_path (path): Path to the input wav file.
        target_eq_raw (path): Path to the target EQ curve.. Defaults to default_target_eq_curve.
        isJson (bool, optional): Flag indicating if target eq is a Json file. Defaults to False.
        verbose (bool, optional): Flag to enable verbose. Defaults to False.

    Returns:
        wav_path_post_eq (path): Path to the output wav file post EQ.
    """    
    wav_path_post_eq = ''.join(wav_path.split('.')[:-1])+'_eq.wav'
    auto_eq_v2(wav_path, wav_path_post_eq, target_eq_raw=target_eq_raw, isJson=isJson, verbose=verbose)
    return wav_path_post_eq
