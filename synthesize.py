from .common import bcolors
from .step1_fluidsynth import mid2wav_withsoundfont as mid2wav, sf_TimbresOfHeaven
from .step2_compressor import compress
from .step3_eq import eq
from .step3_eq import test_auto_eq_v2 as test_eq
from .step3_eq import default_target_eq_curve


def synthesize(
    mid_path, 
    verbose=False, 
    options=[True, True, True], 
    sf=sf_TimbresOfHeaven, 
    target_eq_raw=default_target_eq_curve):
    """XSynthesis' synthesize function.

    Args:
        mid_path (path): The input midi path to the synthesizer.
        verbose (bool, optional): Flag for verbose graphs and logs. Defaults to False.
        options (list, optional): List of 3 booleans corresponding to if each corresponding step will be executed. Defaults to [True, True, True].
        sf (path, optional): Path to the SoundFont for FluidSynth. Defaults to sf_TimbresOfHeaven.
        target_eq_raw (path, optional): Path to the target EQ curve. Defaults to default_target_eq_curve.

    Returns:
        wav_path_post_eq (path): path to the wav file post synthesis.
    """    
    PROC_NAME = f'{bcolors.BOLD}[XSYNTHESIS]{bcolors.ENDC}'
    if options[0]:
        print(PROC_NAME, 'EXPORTING WAV... with', sf)
        wav_path = mid2wav(mid_path, sf)
        print(PROC_NAME, f'EXPORTED WAV: {wav_path}')
    else:
        wav_path = ''.join(mid_path.split('.')[:-1])+'.wav'

    if options[1]:
        print(PROC_NAME, 'APPLYING COMPRESSOR...')
        wav_path_post_compressor = compress(wav_path)
        print(PROC_NAME, f'APPLIED COMPRESSOR: {wav_path_post_compressor}')
    else:
        wav_path_post_compressor = ''.join(mid_path.split('.')[:-1])+'_compressed.wav'

    if options[2]:
        print(PROC_NAME, 'APPLYING EQ...')
        wav_path_post_eq = eq(wav_path_post_compressor, verbose=verbose)
        print(PROC_NAME, f'APPLIED EQ: {wav_path_post_eq}')
        if verbose:
            test_eq(wav_path_post_compressor, wav_path_post_eq, target_eq_raw, isJson=False)
    else:
        wav_path_post_eq = ''.join(mid_path.split('.')[:-1])+'_eq.wav'

    return wav_path_post_eq
