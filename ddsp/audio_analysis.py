import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import pyworld as pw
import parselmouth as pm

import soundfile as sf
import numpy as np

from ddsp.utils import get_mel_fn, get_n_fft
from ddsp.vocoder import Sine_Generator

import matplotlib.pyplot as plt

from typing import Tuple, Union, Optional, Dict

def czt(
        x: torch.Tensor, 
        m: int, 
        A: torch.Tensor, 
        W: torch.Tensor
):
    """
    Args:
        x: tensor [shape = (n)]
        m: int
        A: complex 
        W: complex 

        A = A_0 * exp(j * θ)
        W = W_0 * exp(-j * ϕ)

        通常情况下 A_0 = 1 W_0 = 1 θ = 2π(f0/fs) ϕ = 2π(f0/fs)
    """
    n = x.shape[0]
    l = int(2 ** np.ceil(np.log2(n + m - 1)))
    
    w = W ** (torch.arange(max(m, n), dtype=torch.double, device=x.device)**2 / 2)

    gn = torch.zeros(l, dtype=torch.complex128, device=x.device)
    gn[:n] = (x * (A ** (-torch.arange(0, n, dtype=torch.double,device=x.device))) * w[:n])

    hn = torch.zeros(l, dtype=torch.complex128, device=x.device)
    hn[:m] = 1/w[:m]
    hn[l-n+1:]=1/torch.flip(w[1:n],dims=(0,))

    yk = torch.fft.fft(gn) * torch.fft.fft(hn)
    qn = torch.fft.ifft(yk)
    yn = qn[:m] * w[:m]

    return yn

def sinusoidal_analysis_czt_for(
        audio   : torch.Tensor, 
        f0      : torch.Tensor,
        sr      : int,  
        hop_size: int, 
        max_nhar: int, 
        relative_winsize: int
):
    """
    先用for循环实现，以后优化掉
    Args:
        audio: Tensor [shape = (t)]
        f0:    Tensor [shape = (n_frames)]
        max_nhar: int
        relative_winsize: int
    Returns:
        log_ampl: [shape = (max_nhar, n_frames)]
        x: [shape = (max_nhar, n_frames)]
        phase: [shape = (max_nhar, n_frames)]
    """
    n_frames = f0.shape[0]
    f0 = f0.to(audio.device)

    n_fft, f0_min = get_n_fft(f0, sr, relative_winsize)
    f0 = f0.clamp(min=f0_min)

    nhar = torch.floor(sr / f0 / 2).clamp(max=max_nhar)
    winsize = torch.round(sr / f0 * relative_winsize / 2) * 2
    
    pad = int(n_fft // 2)
    audio_pad  = F.pad(audio, [pad, pad])

    ampl = torch.zeros((max_nhar, n_frames), device=audio.device)
    phase = torch.zeros((max_nhar, n_frames), device=audio.device)
    
    # 先用for循环实现，以后优化掉
    for i in range(n_frames):
        f0_i = f0[i]
        f0_i =  f0_i.to(dtype=torch.double)
        nhar_i = int(nhar[i])
        winsize_i = int(winsize[i])
        start_i = int(i * hop_size)+pad

        window = torch.blackman_window(winsize_i, device=audio.device)
        
        audio_frame = audio_pad[start_i-winsize_i//2 : start_i+winsize_i//2]
        audio_frame = audio_frame * window

        A = torch.exp(torch.complex(torch.tensor(0.,dtype=torch.double,device=audio.device), 2 * np.pi * f0_i/sr))
        W = torch.exp(torch.complex(torch.tensor(0.,dtype=torch.double,device=audio.device), -2 * np.pi * f0_i/sr))

        yn = czt(audio_frame, nhar_i, A, W)
        yn = 2.381 * (yn / (len(audio_frame)//2+1))

        ampl[:int(nhar_i), i] = torch.abs(yn)
        phase[:int(nhar_i), i] = torch.angle(yn)

    return ampl, phase
    
def variable_window_STFT(
        audio: torch.Tensor, 
        n_fft: int, 
        hop_size: int, 
        window_size: torch.Tensor
):
    '''
    window_size可变的STFT
    Args:
        audio: [shape = (t)]
        window_size: Tensor [shape = (n_frames)]
    Returns:
        S: [shape = (n_fft//2+1, n_frames)]
    '''
    pad = int(n_fft // 2)                                                      # n_frames = t//hop_size + 1
    audio_unfold  = F.pad(audio, [pad, pad]).unfold(0, n_fft, hop_size)        # (n_frames, n_fft) 

    window_tensor = generate_window_tensor(window_size, n_fft)                 # (n_fft, n_frames)
    audio_unfold  = audio_unfold * window_tensor.T                             # (n_frames, n_fft)
    S = torch.fft.rfft(audio_unfold).T                                         # (n_fft//2+1, n_frames)

    return S

def generate_window_tensor(window_size, n_fft):
    n_frames = window_size.shape[0]
    window_tensor = torch.zeros((n_fft, n_frames))
    
    for i in range(n_frames):
        winsize = int(window_size[i])
        window = torch.blackman_window(winsize)
        pad_size = (n_fft - winsize) // 2
        window_tensor[pad_size:pad_size + winsize, i] = window
    
    return window_tensor

def sinusoidal_analysis_qifft(
        audio: torch.Tensor, 
        sr: int, 
        hop_size: int, 
        f0: torch.Tensor, 
        max_nhar: int, 
        relative_winsize: int,
        standard_winsize = 1024
):
    # 有bug，不要使用
    n_frames = f0.shape[0]

    n_fft, f0_min = get_n_fft(f0, sr, relative_winsize)
    f0 = f0.clamp(min=f0_min)

    winsize_size = torch.round((sr / f0 * relative_winsize / 2) * 2).int()   # (n_frames)

    standard_window = torch.blackman_window(standard_winsize)
    standard_normalizer = 0.5 * torch.sum(standard_window)
    normalizer = standard_winsize / standard_normalizer / winsize_size       # (n_frames)

    S = variable_window_STFT(
        audio, n_fft, hop_size, winsize_size
    ) * normalizer                           # (n_fft//2+1, n_frames)
    
    spec_magn, spec_phse = torch.abs(S), torch.angle(S)
    log_spec_magn = torch.log(torch.clamp(spec_magn, min=1e-8))

    qifft_tensor   = torch.zeros((3, max_nhar, n_frames), device=audio.device)
    peak_bin_tensor   = torch.zeros((max_nhar, n_frames), device=audio.device)
    remove_above_nhar = torch.zeros((max_nhar, n_frames), device=audio.device)

    tolerance = 0.3
    nhars = torch.clamp((sr / (f0 * 2)).floor().int(), max=max_nhar)
    f0_proportions = f0 / sr * n_fft

    for i in range(n_frames):
        nhar, f0_proportion = nhars[i], f0_proportions[i]
        remove_above_nhar[:nhar, i] = 1

        l_idxs = (
            (torch.arange(
                1,nhar+1, device=audio.device) - tolerance
            ) * f0_proportion
        ).round().clamp(1, n_fft//2 - 1)

        u_idxs = (
            (torch.arange(
                1,nhar+1, device=audio.device) + tolerance
            ) * f0_proportion
        ).round().clamp(1, n_fft//2 - 1)

        for j in range(nhar):
            l_idx, u_idx = int(l_idxs[j]), int(u_idxs[j])
            peak_bin = torch.argmax(log_spec_magn[l_idx:u_idx+1, i])
            peak_bin += l_idx
            peak_bin_tensor[j, i] = peak_bin

            qifft_tensor[0, j, i] = log_spec_magn[peak_bin - 1, i]
            qifft_tensor[1, j, i] = log_spec_magn[peak_bin    , i]
            qifft_tensor[2, j, i] = log_spec_magn[peak_bin + 1, i]

    log_ampl, x = qifft(qifft_tensor)
    ampl = torch.exp(log_ampl)
    phase = torch.zeros((max_nhar, n_frames), device=audio.device)

    x = x + peak_bin_tensor
    interp_x = np.linspace(0, n_fft//2, n_fft//2 + 1)
    for i in range(n_frames):
        phase[:, i] = torch.from_numpy(
            np.interp(
                x[:, i].numpy(), 
                interp_x, 
                np.unwrap(spec_phse[:, i].numpy())
            )
        )                                

    ampl = ampl * remove_above_nhar
    phase = phase * remove_above_nhar

    return ampl, phase
    
def qifft(qifft_tensor: torch.Tensor):
    '''
    Args:
        qifft_tensor: (3, max_nhar, n_frames)
    '''
    a = qifft_tensor[0, :, :]          # (max_nhar, n_frames)
    b = qifft_tensor[1, :, :]
    c = qifft_tensor[2, :, :]

    a1 = (a + c) / 2.0 - b             # (max_nhar, n_frames)
    a2 =  c  -  b  - a1                   
    x  = -a2 / (a1 + 1e-8) * 0.5             
    x[torch.abs(x) > 1] = 0           

    ret = a1 * x * x + a2 * x + b      # (max_nhar, n_frames)
    idx = ret > b + 0.2
    ret[idx] = b[idx] + 0.2            # Why? I don't know.
    return ret, x

class SinusoidalAnalyzer(nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            hop_size     : int,
            max_nhar     : int,
            relative_winsize: int,
            device       : str = 'cpu',
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.max_nhar = max_nhar
        self.relative_winsize = relative_winsize
        self.device = device

    def forward(
            self,
            x : torch.Tensor,
            f0 : torch.Tensor,
            model : str = 'czt' #目前只能用czt
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze the given audio signal to extract sinusoidal parameters.
        
        Args:
            x (torch.Tensor): Audio signal tensor of shape (t,).
            f0 (torch.Tensor): F0 tensor of shape (n_frames,).
            n_frames (int): Number of frames, equal to the length of mel_spec.
            model (str): Type of sinusoidal analysis model ('czt' or 'qifft').
        
        Returns:
            torch.Tensor: Extracted sinusoidal parameters of shape (max_nhar, n_frames).
        """
        if model == 'czt':
            ampl, phase = sinusoidal_analysis_czt_for(
                x, f0, self.sampling_rate, self.hop_size, self.max_nhar, self.relative_winsize
            )
        elif model == 'qifft':
            ampl, phase = sinusoidal_analysis_qifft(
                x, self.sampling_rate, self.hop_size, f0, self.max_nhar, self.relative_winsize
            )
        else:
            raise ValueError(f" [x] Unknown sinusoidal analysis model: {model}")

        return ampl, phase
    

    
class F0Analyzer(nn.Module):
    def __init__(
            self, 
            sampling_rate: int, 
            f0_extractor : str, 
            hop_size     : int,
            f0_min       : float, 
            f0_max       : float,
    ):
        """  
        Args:
            sampling_rate (int): Sampling rate of the audio signal.
            f0_extractor (str): Type of F0 extractor ('parselmouth', 'dio', or 'harvest').
            f0_min (float): Minimum F0 in Hz.
            f0_max (float): Maximum F0 in Hz.
            hop_size (int): Hop size in samples.
        """
        super(F0Analyzer, self).__init__()
        self.sampling_rate = sampling_rate
        self.f0_extractor  = f0_extractor
        self.hop_size      = hop_size
        self.f0_min        = f0_min
        self.f0_max        = f0_max

    def forward(
            self, 
            x: torch.Tensor, 
            n_frames: int
    ) -> torch.Tensor:
        """
        Analyze the given audio signal to extract F0.
        
        Args:
            x (torch.Tensor): Audio signal tensor of shape (t,).
            n_frames (int): Number of frames, equal to the length of mel_spec.
        
        Returns:
            torch.Tensor: Extracted F0 of shape (n_frames,).
        """
        x = x.to('cpu').numpy()

        if self.f0_extractor == 'parselmouth':
            f0 = self._extract_f0_parselmouth(x, n_frames)
        elif self.f0_extractor == 'dio':
            f0 = self._extract_f0_dio(x, n_frames)
        elif self.f0_extractor == 'harvest':
            f0 = self._extract_f0_harvest(x, n_frames)
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
        
        uv = f0 == 0
        return f0, uv

    def _extract_f0_parselmouth(self, x: np.ndarray, n_frames):
        l_pad = int(
                np.ceil(
                    1.5 / self.f0_min * self.sampling_rate
                )
        )
        r_pad = self.hop_size * ((len(x) - 1) // self.hop_size + 1) - len(x) + l_pad + 1
        padded_signal = np.pad(x, (l_pad, r_pad))
        
        sound = pm.Sound(padded_signal, self.sampling_rate)
        pitch = sound.to_pitch_ac(
            time_step=self.hop_size / self.sampling_rate, 
            voicing_threshold=0.6,
            pitch_floor=self.f0_min, 
            pitch_ceiling=1100
        )
        
        f0 = pitch.selected_array['frequency']
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        f0 = f0[:n_frames]

        return f0

    def _extract_f0_dio(self, x: np.ndarray, n_frames: int) -> np.ndarray:
        _f0, t = pw.dio(
            x.astype('double'), 
            self.sampling_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            channels_in_octave=2, 
            frame_period=(1000 * self.hop_size / self.sampling_rate)
        )
        
        f0 = pw.stonemask(x.astype('double'), _f0, t, self.sampling_rate)
        return f0.astype('float')[:n_frames]

    def _extract_f0_harvest(self, x: np.ndarray, n_frames: int) -> np.ndarray:
        f0, _ = pw.harvest(
            x.astype('double'), 
            self.sampling_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            frame_period=(1000 * self.hop_size / self.sampling_rate)
        )
        return f0.astype('float')[:n_frames]
    
    
class MelAnalysis(nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            win_size     : int,
            hop_size     : int,
            n_mels       : int,
            n_fft        : Optional[int] = None,
            mel_fmin     : float = 0.0,
            mel_fmax     : Optional[float] = None,
            clamp        : float = 1e-5,
            device       : str = 'cpu',
    ):
        super().__init__()
        n_fft = win_size if n_fft is None else n_fft
        self.hann_window: Dict[str, torch.Tensor] = {}

        mel_basis = get_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
            device=device,
        )
        mel_basis = mel_basis.float()
        self.register_buffer("mel_basis", mel_basis)

        self.hop_size: int   = hop_size
        self.win_size: int   = win_size
        self.n_fft   : int   = n_fft
        self.clamp   : float = clamp

    def _get_hann_window(self, win_size: int, device) -> torch.Tensor:
        key: str = f"{win_size}_{device}"
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(win_size).to(device)
        return self.hann_window[key]

    def _get_mel(
            self, 
            audio: torch.Tensor, 
            keyshift: float = 0.0, 
            speed: float = 1.0, 
            diffsinger: bool = True
    ) -> torch.Tensor:
        factor: float = 2 ** (keyshift / 12)
        n_fft_new: int = int(np.round(self.n_fft * factor))
        win_size_new: int = int(np.round(self.win_size * factor))
        hop_size_new: int = int(np.round(self.hop_size * speed))
        hann_window_new: torch.Tensor = self._get_hann_window(win_size_new, audio.device)

        # 处理双声道信号
        if len(audio.shape) == 2:
            print("双声道信号")
            audio = audio[:, 0]

        if diffsinger:
            print(audio.shape)
            audio = F.pad(
                audio.unsqueeze(0),
                ((win_size_new - hop_size_new) // 2, (win_size_new - hop_size_new + 1) // 2),
                mode='reflect'
            ).squeeze(0)
            print(audio.shape)
            center: bool = False
        else:
            center = True

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_size_new,
            win_length=win_size_new,
            window=hann_window_new,
            center=center,
            return_complex=True
        )
        magnitude: torch.Tensor = fft.abs()

        if keyshift != 0:
            size: int = self.n_fft // 2 + 1
            resize: int = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_size / win_size_new

        mel_output: torch.Tensor = torch.matmul(self.mel_basis, magnitude)
        return mel_output

    def forward(
            self, 
            audio: torch.Tensor, 
            keyshift: float = 0.0, 
            speed: float = 1.0, 
            diffsinger:bool = True,
            mel_base: str = 'e'
    ) -> torch.Tensor:
        if torch.min(audio) < -1.0 or torch.max(audio) > 1.0:
            print('Audio values exceed [-1., 1.] range')

        mel: torch.Tensor = self._get_mel(audio, keyshift=keyshift, speed=speed, diffsinger=diffsinger)
        print(self.clamp)
        log_mel_spec: torch.Tensor = torch.log(torch.clamp(mel, min=self.clamp))

        if mel_base != 'e':
            assert mel_base in ['10', 10], "mel_base must be 'e', '10' or 10."
            log_mel_spec *= 0.434294  # Convert log to base 10

        # mel shape: (n_mels, n_frames)

        return log_mel_spec

if __name__ == '__main__':
    # test
    from utils import interp_f0

    audiopath = r'E:/pc-ddsp5.29/TheKitchenettes_Alive_RAW_08_030.wav'
    audio=sf.read(audiopath)[0]
    audio= torch.from_numpy(audio).float()

    sampling_rate = 44100
    hop_size = 512
    n_fft = 2048
    n_mels = 128

    mel_extractor = MelAnalysis(
        sampling_rate=sampling_rate,
        win_size=n_fft,
        hop_size=hop_size,
        n_mels=n_mels,
        n_fft=n_fft,
        mel_fmin=40,
        mel_fmax=16000,
        clamp=1e-6,
        device='cpu',
    )

    mel=mel_extractor(audio, keyshift=0.0, speed=1.0, diffsinger=True, mel_base='e')
    print(mel.shape)
    nfreams = mel.shape[1]
    print(nfreams)
    print(audio.shape)
    print(audio.shape[0]//hop_size)

    f0_analyzer = F0Analyzer(
        sampling_rate=sampling_rate,
        f0_extractor='parselmouth',
        hop_size=hop_size, 
        f0_min=40, 
        f0_max=800,
    )

    f0, uv = f0_analyzer(audio, n_frames=nfreams)

    print(f0.shape)
    f0, _  = interp_f0(f0, uv)


    ampl, phase = sinusoidal_analysis_czt_for(
        audio=audio, 
        sr=sampling_rate, 
        hop_size=hop_size, 
        f0=torch.from_numpy(f0), 
        max_nhar=256,
        relative_winsize=4, 
    )

    vocoder = Sine_Generator(hop_size=hop_size, sampling_rate=sampling_rate, device='cpu')
    y = vocoder(ampl.unsqueeze(0), phase.unsqueeze(0), torch.from_numpy(f0).unsqueeze(0))

    print(y.shape)

    sf.write('test.wav', y.numpy(), sampling_rate,subtype='FLOAT')


