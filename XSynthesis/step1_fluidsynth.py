from midi2audio import FluidSynth

from .common import bcolors

PROC_NAME = f'{bcolors.BOLD}[FLUIDSYNTH]{bcolors.ENDC}'

def mid2wav_withsoundfont(mid_path: str, sf_path: str) -> str:
    """Exports midi to WAV file using a selected SoundFont.

    Args:
        mid_path (path): Input midi path.
        sf_path (path): Input SoundFont path.

    Returns:
        wav_path (path): Output wav path from FluidSynth. 
    """
    print(PROC_NAME, f'Loading SoundFont from path: {sf_path}')
    fs = FluidSynth(sound_font=sf_path)
    wav_path = ''.join(mid_path.split('.')[:-1])+'.wav'
    print(PROC_NAME, f'Reading from path: {mid_path}')
    fs.midi_to_audio(mid_path, wav_path) 
    print(PROC_NAME, f'Writting path: {wav_path}')
    return wav_path

# NOTE: Timbres of Heavens soundfont path, change this to environment variable for production!
sf_TimbresOfHeaven = 'XSynthesis/configurations/TimbresOfHeaven.sf2'

