import os
import soundfile as sf
import numpy as np
import click
import torch

from ddsp.audio_analysis import F0Analyzer, SinusoidalAnalyzer
from ddsp.vocoder import Sine_Generator, Sine_Generator_Fast

@click.command()
@click.option('--input_file', type=click.Path(exists=True))
def main(input_file):
    # Load input file
    audio, sr = sf.read(input_file)
    audio= torch.from_numpy(audio).float()
    hop_size = 512
    sin_mag = 128

    f0, uv = F0Analyzer(sampling_rate = sr, 
                    f0_extractor = 'parselmouth', 
                    hop_size = hop_size, 
                    f0_min = 30, 
                    f0_max = 800)(audio, len(audio)//hop_size)
    f0=torch.from_numpy(f0).float()
    f0_frames = f0.unsqueeze(0).unsqueeze(1)     # [1 x 1 x n_frames]

    
    nhar_range = torch.arange(
        start = 1, end = sin_mag + 1).unsqueeze(0).unsqueeze(-1)   # [max_nhar] -> [1, max_nhar, 1]
    f0_list = f0_frames * nhar_range  # [1 x max_nhar x n_frames]

    ampl, phase = SinusoidalAnalyzer(sampling_rate = sr,
                                    hop_size = hop_size,
                                    max_nhar = sin_mag,
                                    relative_winsize = 4
                                    )(
                                    audio, 
                                    f0=f0, 
                                )

    harmonic_audio = Sine_Generator_Fast(hop_size = hop_size,sampling_rate = sr)(ampl.unsqueeze(0), phase.unsqueeze(0), f0_list)
    harmonic_audio = harmonic_audio.squeeze(0)

    harmonic_audio = torch.nn.functional.pad(  #把长度补全到audio的长度
        harmonic_audio, (0, audio.shape[0] - harmonic_audio.shape[0]), 'constant', 0)

    noise_audio = audio - harmonic_audio

    # Save output file
    harmonic_audio_file = os.path.splitext(input_file)[0] + '_harmonic.wav'
    sf.write(harmonic_audio_file, harmonic_audio.numpy(), sr)

    noise_audio_file = os.path.splitext(input_file)[0] + '_noise.wav'
    sf.write(noise_audio_file, noise_audio.numpy(), sr)




if __name__ == '__main__':
    main()