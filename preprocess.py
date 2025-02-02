
import numpy as np
import torch


from tqdm import tqdm
from logger.utils import DotDict, traverse_dir, load_config
import soundfile as sf
import click
from pathlib import Path
from ddsp.audio_analysis import MelAnalysis, F0Analyzer, SinusoidalAnalyzer
from ddsp.utils import interp_f0, expand_uv

class Preprocessor:
    def __init__(self, config: DotDict, device: str):
        self.config = config

        self.sampling_rate = config.data.sampling_rate
        self.train_path = Path(config.data.train_path)
        self.valid_path = Path(config.data.valid_path)

        self.device = device
        print(f'Preprocessor using device: {self.device}')
 
        self.mel_extractor = MelAnalysis(
            sampling_rate = self.sampling_rate,
            hop_size = config.data.hop_size,
            win_size = config.data.win_size,
            n_fft = config.data.n_fft,
            n_mels = config.data.n_mels,
            mel_fmin = config.data.mel_fmin,
            mel_fmax = config.data.mel_fmax,
            clamp  = config.data.mel_clamp,
            device = self.device
        )

        self.f0_extractor = F0Analyzer(
            sampling_rate = self.sampling_rate,
            f0_extractor = config.data.f0_extractor,
            hop_size = config.data.hop_size,
            f0_min = config.data.f0_min,
            f0_max = config.data.f0_max,
        )

        self.sinusoidal_analyzer = SinusoidalAnalyzer(
            sampling_rate = self.sampling_rate,
            hop_size = config.data.hop_size,
            max_nhar = config.data.max_nhar,
            relative_winsize = config.data.relative_winsize,
            device = self.device
        )

    def __call__(self):
        return self.preprocess()

    def preprocess(self):
        for base_path in [self.train_path, self.valid_path]:
            # list files
            filelist = traverse_dir(
                base_path / "audio",
                extension="wav",
                is_pure=True,
                is_sort=True,
                is_ext=False,
            )

            for file in tqdm(filelist):
                path_harmonic_audio = base_path / "harmonic_audio" / f'{file}.wav'
                path_audio = base_path / "audio" / f'{file}.wav'
                path_phase = base_path / "phase" / f'{file}.npy'
                path_ampl = base_path / "ampl"  / f'{file}.npy'
                path_mel = base_path / "mel"  / f'{file}.npy'
                path_f0  = base_path / "f0"  / f'{file}.npy'
                path_uv = base_path / "uv" / f'{file}.npy'

                # load audio 加载音频
                audio, sr =  sf.read(str(path_audio))
                assert sr == self.sampling_rate, f'Sampling rate of {path_audio} is not {self.sampling_rate}'
                audio = torch.from_numpy(audio).float().to(self.device)

                haudio, sr = sf.read(str(path_harmonic_audio))
                assert sr == self.sampling_rate, f'Sampling rate of {path_harmonic_audio} is not {self.sampling_rate}'
                haudio     = torch.from_numpy(haudio).float().to(self.device)

                try:
                    assert audio.shape[0] == haudio.shape[0]
                    # extract mel, f0, uv 特征提取
                    mel, f0, uv = self.mel_f0_uv_process(audio)

                    # extract amplitude and phase 振幅和相位分析
                    tf0 = torch.from_numpy(f0).float().to(self.device)
                    ampl, phase = self.ampl_phase_process(haudio, tf0)
                except:
                    Path(path_audio).unlink(missing_ok=True)
                    tqdm.write(f'Audio file {path_audio} f0 extraction failed. Deleted.')
                    continue
                
                # 创建空文件
                path_mel.parent.mkdir(parents=True, exist_ok=True)
                path_f0.parent.mkdir(parents=True, exist_ok=True)
                path_uv.parent.mkdir(parents=True, exist_ok=True)
                path_phase.parent.mkdir(parents=True, exist_ok=True)
                path_ampl.parent.mkdir(parents=True, exist_ok=True)
                
                # save npy 保存女朋友
                np.save(path_mel, mel)
                np.save(path_f0 , f0 )
                np.save(path_uv , uv )
                np.save(path_phase, phase)
                np.save(path_ampl, ampl)
            
    def mel_f0_uv_process(self, audio: torch.Tensor):
        # extract mel 特征提取
        mel = self.mel_extractor(audio, diffsinger=True)
        mel = mel.to('cpu').numpy()

        # extract f0 and uv 基频分析
        f0, uv = self.f0_extractor(audio, n_frames=mel.shape[1])

        f0, _  = interp_f0(f0, uv)
        uv     = expand_uv(uv)
    
        return mel, f0, uv
    
    def ampl_phase_process(self, audio: torch.Tensor, f0 : torch.Tensor):
        # extract amplitude and phase 振幅和相位分析
        ampl, phase = self.sinusoidal_analyzer(audio, f0, model = 'czt')
        ampl = ampl.to('cpu').numpy()
        phase = phase.to('cpu').numpy()

        return ampl, phase
    


@click.command(help='Preprocess audio files')
@click.option(
    '--config', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=Path, resolve_path=True
    ),
    required=True, metavar='CONFIG_FILE',
    help='The path to the config file.'
)
@click.option(
    '--device', type=str, default=None,
    help='The device to use for preprocessing.'
)
def main(config, device):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Preprocessor using device: {device}')

    # load config
    args = load_config(config)

    # TODO: add config validation
    # validate_config(args)

    preprocessor = Preprocessor(args, device)
    preprocessor()

if __name__ == '__main__':
    main()