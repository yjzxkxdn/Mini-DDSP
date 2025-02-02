import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ddsp.mel2control import Mel2Control
from .utils import upsample

def compute_inphase(f0_sum, hop_size, sampling_rate, device, inference=False):
    '''
    Args:
        f0_sum: [shape = (batch, C, T-1)]
        hop_size: int
        sampling_rate: int
        B: int
        max_nhar: int
        device: str
        inference: bool, default False
    Returns:
        inphase: [shape = (B, C, T)]
    '''
    B, C, _  = f0_sum.shape
    if inference:
        inphase = torch.cumsum(
            (f0_sum * np.pi / sampling_rate * hop_size).double()%(2*np.pi), dim=2
        )
    else:
        inphase = torch.cumsum(
            (f0_sum * np.pi / sampling_rate * hop_size)%(2*np.pi), dim=2
        )

    inphase = torch.cat(
        (torch.zeros(B, C, 1).to(device), inphase), dim=2
    ) % (2*np.pi)                                                 
    return inphase  # [batch, C, T]

def replicate(t, x, batch):
    """
    Replicates tensor t to have length x.
    Args:
        t: input tensor [batch, channels, time]
        x: output tensor length
        batch: int
    Returns:
        replicated: tensor [batch, channels, x]
    """
    repeat_times = (x + t.size(-1) - 1) // t.size(-1)
    replicated = t.repeat(batch, 1, repeat_times)
    replicated = replicated[:,:,:x]
    return replicated

def get_remove_above_fmax(n_harm, pitch, fmax, level_start=1):
    '''
    Args:
        pitch: b x t x 1
        fmax: float
        level_start: int, default 1
    Returns:
        aa: b x t x n_harm
    '''
    pitches = pitch * torch.arange(level_start, n_harm + level_start).to(pitch)
    rm = (pitches < fmax).float() + 1e-7
    return rm

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.unsafe_load(config)
    args = DotDict(args)
    
    model_path = Path(model_path)  
    # load model
    print(' [Loading] ' + str(model_path))
    if model_path.suffix == '.jit':
        model = torch.jit.load(model_path, map_location=torch.device(device))
    else:
        if args.model.type == 'SinStack':
            model = SinStack(
                args,
                device=device)
        else:
            raise ValueError(f" [x] Unknown Model: {args.model.type}")
        model.to(device)
        ckpt = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        model.eval()
    return model, args
        
class SinStack(torch.nn.Module):
    def __init__(self, 
            args,
            device='cuda'):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(args.data.sampling_rate))
        self.register_buffer("hop_size", torch.tensor(args.data.hop_size))
        self.register_buffer("win_length", torch.tensor(2*args.data.hop_size))
        self.register_buffer("window", torch.hann_window(2*args.data.hop_size))
        self.register_buffer("sin_mag", torch.tensor(args.model.n_sin_hars))
        self.register_buffer("noise_mag", torch.tensor(args.model.n_noise_bin))
        self.register_buffer("uv_noise_k", torch.tensor(args.model.uv_noise_k))

        # Mel2Control
        split_map = {            
            'sin_mag'  : args.model.n_sin_hars,
            'sin_phase': args.model.n_sin_hars,
            'noise_mag': args.model.n_noise_bin,
        }

        self.register_buffer("u_noise", \
                             torch.load('u_noise.ckpt', map_location=torch.device(device)))
        self.register_buffer("v_noise", \
                             torch.load('v_noise.ckpt', map_location=torch.device(device)))
        
        print(' [DDSP Model] Mel2Control',self.v_noise.shape)
        
        self.mel2ctrl = Mel2Control(
            args.data.n_mels, 
            args.model.n_sin_hars, 
            args.data.hop_size, 
            split_map
        )
        self.sine_generator = Sine_Generator(
            args.data.hop_size, 
            args.data.sampling_rate, 
            device=device
        )
        self.noise_generator = Noise_Generator(
            args.data.sampling_rate,
            args.data.hop_size,
            self.v_noise,
            self.u_noise,
            device=device,
            triangle_ReLU = args.model.triangle_ReLU,
            triangle_ReLU_up = args.model.triangle_ReLU_up,
            triangle_ReLU_down = args.model.triangle_ReLU_down,
        )

        self.device = device

    def phase_prediction(self, phase_pre_model, sin_phase, f0_list,inference):
        '''
        Args:
            f0_list: [shape = (batch, max_nhar, T)]
            sin_phase: [shape = (batch, max_nhar, T)]
            inference: bool, default False
            
        Returns:
            sin_phase: [shape = (batch, max_nhar, T)]
        '''
        if phase_pre_model == 'offset':
            f0_sum = f0_list[:, :, 1:] + f0_list[:, :, :-1]  # [batch, max_nhar, T-1]
            inphase = compute_inphase(
                f0_sum, 
                self.hop_size, 
                self.sampling_rate, 
                self.device, 
                inference=inference
            )
            sin_phase = inphase + sin_phase # [batch, max_nhar, T]
        elif phase_pre_model == 'adjacent difference':
            f0_sum = (f0_list[:, 0, 1:] + f0_list[:, 0, :-1]).unsqueeze(1) # [batch, 1, T-1]
            inphase = compute_inphase(
                f0_sum, 
                self.hop_size, 
                self.sampling_rate, 
                self.device, 
                inference=inference
            )
            sin_phase[:, 0, :] = sin_phase[:, 0, :] + inphase.squeeze(1) # 为基频添加初始相位
            sin_phase = torch.cumsum(sin_phase, dim=1) # [batch, max_nhar, T]
        elif phase_pre_model == 'fundamental difference':
            f0_sum = (f0_list[:, 0, 1:] + f0_list[:, 0, :-1]).unsqueeze(1) # [batch, 1, T-1]
            inphase = compute_inphase(
                f0_sum, 
                self.hop_size, 
                self.sampling_rate, 
                self.device, 
                inference=inference
            )
            sin_phase[:, 0, :] = sin_phase[:, 0, :] + inphase.squeeze(1) # 为基频添加初始相位
            sin_phase[:, 1:, :] = sin_phase[:, 1:, :] + sin_phase[:, 0, :].unsqueeze(1) 
        elif phase_pre_model == 'absolute position':
            pass
        else:
            raise ValueError(f" [x] Unknown phase_pre_model: {phase_pre_model}")
        
        return sin_phase

            
    def forward(
            self, 
            mel_frames,
            f0_frames,
            inference=False,
            phase_pre_model='offset', # 'offset','absolute position' or 'difference'
            **kwargs
    ):
        '''
            mel_frames: B x n_mels x n_frames
            f0_frames: B x n_frames x 1
        '''    
        nhar_range = torch.arange(
            start = 1, end = self.sin_mag + 1, device=self.device
        ).unsqueeze(0).unsqueeze(-1)                              # [max_nhar] -> [1, max_nhar, 1]

        f0_list = f0_frames.unsqueeze(1).squeeze(3) * nhar_range  # [batch, 1, T] * [1, max_nhar, 1] -> [batch, max_nhar, T]
        inp = self.phase_prediction(phase_pre_model, torch.tensor(0.0), f0_list, inference)

        inp = inp.to(torch.float32)

        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames.transpose(1, 2), inp.transpose(1, 2))

        sin_mag   = torch.exp(ctrls['sin_mag']) / 128    # b x T x max_nhar
        sin_phase = ctrls['sin_phase']                   # b x T x max_nhar
        noise_mag = torch.exp(ctrls['noise_mag']) / 128  # b x T x n_noise

        # permutation
        sin_mag   = sin_mag.permute(0, 2, 1)     # b x max_nhar x T
        sin_phase = sin_phase.permute(0, 2, 1)   # b x max_nhar x T
        noise_mag = noise_mag.permute(0, 2, 1)   # b x n_noise x T
        B, max_nhar, T = sin_mag.shape

        # remove above fmax
        rm_mask = get_remove_above_fmax(
            max_nhar, 
            f0_frames,
            fmax = self.sampling_rate / 2
        ).permute(0, 2, 1)
        # sin_phase = self.phase_prediction(phase_pre_model, sin_phase, f0_list, inference)
        sin_phase = sin_phase + inp # 当使用offset时，不需要算两遍初始相位。使用其它模式时需要注释掉这行使用上面那一行
        sin_mag, sin_phase = rm_mask * sin_mag, rm_mask * sin_phase

        harmonic = self.sine_generator(sin_mag, sin_phase, f0_list)

        # get uv mask
        noise_mag_total = noise_mag.sum(dim=1) # b x T
        harmonic_mag_first = sin_mag[:, 0, :] # b x T
        uv_mask = self.uv_noise_k *harmonic_mag_first / (self.uv_noise_k*harmonic_mag_first + noise_mag_total + 1e-7) # b x T

        # noise generation
        noise = self.noise_generator(noise_mag, uv_mask, f0_frames.permute(0, 2, 1))
                       
        signal = harmonic + noise  # [batch, T*hop_size]

        return signal, 0, (harmonic, noise), (sin_mag, sin_phase)
    
    

class Sine_Generator(torch.nn.Module):
    def __init__(self, hop_size, sampling_rate, device='cpu'):
        super().__init__()
        self.hop_size = hop_size
        self.win_size = hop_size * 2
        self.window = torch.hann_window(self.win_size).to(device)
        self.sampling_rate = sampling_rate
        self.device = device

    def forward(self, ampl, phase, f0_list, inference=False):
        '''
            ampl: B x max_nhar x T
            phase: B x max_nhar x T
            f0_list: B x max_nhar x T
        '''  
        B, _, T = ampl.shape
        x_list = torch.arange(-self.hop_size, self.hop_size).to(self.device)
        x_list = x_list.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 x 1 x 1 x win_size

        freq_list = (2 * np.pi * f0_list / self.sampling_rate).unsqueeze(-1) * x_list + phase.unsqueeze(-1)  # [batch, max_nhar, T, win_size]

        y_tmp = torch.cos(freq_list) * ampl.unsqueeze(-1)  # [batch, max_nhar, T, win_size]
        y_tmp = y_tmp.sum(dim=1)  # [batch, T, win_size]

        hann_window = self.window.unsqueeze(0).unsqueeze(0)  # [1, 1, win_size]
        y_tmp_weighted = y_tmp * hann_window  # [batch, T, win_size]

        '''# 将 y_tmp_weighted 转换为适合 fold 的形状
        y_tmp_weighted_reshaped = y_tmp_weighted.permute(0, 2, 1).contiguous()
        y_tmp_weighted_reshaped = y_tmp_weighted_reshaped.view(B * self.win_size, T) 
        y_tmp_weighted = y_tmp * hann_window  # [batch, T, win_size]
        # 使用 fold 函数来实现滑动窗口效果
        output = torch.nn.functional.fold(
            y_tmp_weighted_reshaped.unsqueeze(0), 
            output_size=(T * self.hop_size + self.hop_size, 1), 
            kernel_size=(self.win_size, 1), 
            stride=(self.hop_size, 1)
        )
        output = output[:, :, self.hop_size:]

        y_return = output.squeeze(0).view(B, T * self.hop_size)'''

        y_tmp_padded = F.pad(y_tmp_weighted , (0, 0, 0,T % 2), "constant", 0)
        new_T = y_tmp_padded.shape[1]

        y_tmp_reshaped = y_tmp_padded.view(B, new_T//2, 2, self.win_size)
        tensor_even = y_tmp_reshaped[:, :, 0, :]  # 偶数时间步
        tensor_odd = y_tmp_reshaped[:, :, 1, :]   # 奇数时间步

        tensor_even = tensor_even.reshape(B, (new_T//2)*self.win_size)
        tensor_odd = tensor_odd.reshape(B, (new_T//2)*self.win_size)

        tensor_even = F.pad(tensor_even, (0, self.hop_size), "constant", 0)
        tensor_odd = F.pad(tensor_odd, (self.hop_size, 0), "constant", 0)
        cat_tensor = torch.cat((tensor_even.unsqueeze(-1), tensor_odd.unsqueeze(-1)), dim=2)
        sum_tensor = torch.sum(cat_tensor, dim=2)
        y_return = sum_tensor[:, self.hop_size:T*self.hop_size + self.hop_size]

        return y_return # [batch, T*hop_size]
    
class Sine_Generator_Fast(torch.nn.Module):
    def __init__(self, hop_size, sampling_rate, device='cpu'):
        super().__init__()
        self.hop_size = hop_size
        self.win_size = hop_size * 2
        self.window = torch.hann_window(self.win_size).to(device)
        self.sampling_rate = sampling_rate
        self.device = device

    def forward(self, ampl, phase, f0_list, inference=False):
        '''
            ampl: B x max_nhar x T
            phase: B x max_nhar x T
            f0_list: B x max_nhar x T
        ''' 
        B, max_nhar, T = ampl.shape
        
        k = 16 
        winsize = self.win_size
        x_start_list = torch.arange(-winsize/2, winsize/2, winsize/k, device=self.device)
        x_start_list_c = x_start_list.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 x 1 x 1 x k

        x_start_list = (x_start_list+ winsize/2).to(torch.int)

        omega = 2 * np.pi * f0_list / self.sampling_rate  # [batch, max_nhar, T]
        omega = omega.unsqueeze(-1) # [batch, max_nhar, T, 1]

        ampl = ampl.unsqueeze(-1) # [batch, max_nhar, T, 1]
        phase = phase.unsqueeze(-1) # [batch, max_nhar, T, 1]
        
        c = 2 * torch.cos(omega) # [batch, max_nhar, T, 1]
        y_tmp = torch.zeros(B, max_nhar, T, winsize, device=self.device)
        y_tmp[:,:,:,x_start_list] = ampl * torch.cos(omega * x_start_list_c + phase)  # [batch, max_nhar, T, k]
        y_tmp[:,:,:,x_start_list+1] = ampl * torch.cos(omega * (x_start_list_c + 1) + phase)  # [batch, max_nhar, T, k]
        

        for i in range(2, int(winsize/k)):
            y_tmp[:,:,:,x_start_list+i] = c * y_tmp[:,:,:,x_start_list+i-1] - y_tmp[:,:,:,x_start_list+i-2]

        # y_tmp = torch.cos(freq_list) * ampl.unsqueeze(-1)  # [batch, max_nhar, T, win_size]
        y_tmp = y_tmp.sum(dim=1)  # [batch, T, win_size]

        hann_window = self.window.unsqueeze(0).unsqueeze(0)  # [1, 1, win_size]

        y_tmp_weighted = y_tmp * hann_window  # [batch, T, win_size]

        '''
        # 将 y_tmp_weighted 转换为适合 fold 的形状
        y_tmp_weighted_reshaped = y_tmp_weighted.permute(0, 2, 1).contiguous()
        y_tmp_weighted_reshaped = y_tmp_weighted_reshaped.view(B * self.win_size, T) 

        # 使用 fold 函数来实现滑动窗口效果
        output = torch.nn.functional.fold(
            y_tmp_weighted_reshaped.unsqueeze(0), 
            output_size=(T * self.hop_size + self.hop_size, 1), 
            kernel_size=(self.win_size, 1), 
            stride=(self.hop_size, 1)
        )
        output = output[:, :, self.hop_size:]

        y_return = output.squeeze(0).view(B, T * self.hop_size)'''


        y_tmp_padded = F.pad(y_tmp_weighted , (0, 0, 0,T % 2), "constant", 0)
        new_T = y_tmp_padded.shape[1]

        y_tmp_reshaped = y_tmp_padded.view(B, new_T//2, 2, self.win_size)
        tensor_even = y_tmp_reshaped[:, :, 0, :]  # 偶数时间步
        tensor_odd = y_tmp_reshaped[:, :, 1, :]   # 奇数时间步

        tensor_even = tensor_even.reshape(B, (new_T//2)*self.win_size)
        tensor_odd = tensor_odd.reshape(B, (new_T//2)*self.win_size)

        tensor_even = F.pad(tensor_even, (0, self.hop_size), "constant", 0)
        tensor_odd = F.pad(tensor_odd, (self.hop_size, 0), "constant", 0)
        cat_tensor = torch.cat((tensor_even.unsqueeze(-1), tensor_odd.unsqueeze(-1)), dim=2)
        sum_tensor = torch.sum(cat_tensor, dim=2)
        y_return = sum_tensor[:, self.hop_size:T*self.hop_size + self.hop_size]

        return y_return # [batch, T*hop_size]
    

class Noise_Generator(torch.nn.Module):
    
    def __init__(self, sampling_rate, hop_size, v_noise, u_noise, triangle_ReLU = True ,triangle_ReLU_up = 0.2, triangle_ReLU_down = 0.8,device='cpu'):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.device = device
        self.noiseop = v_noise
        self.noiseran = u_noise
        self.triangle_ReLU = triangle_ReLU
        self.triangle_ReLU_up = triangle_ReLU_up
        self.triangle_ReLU_down = triangle_ReLU_down
    
    @staticmethod
    def Triangle_ReLU(x:torch.Tensor, x1,x2):
        '''
        Triangle ReLU activation function
        '''
        return -(torch.relu(-(x-x1))/x1) - (torch.relu(x-x1)/x2) + torch.relu(x-x1-x2)/x2 +1
        
    def fast_phase_gen(self, f0_frames):
        n = torch.arange(self.hop_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.hop_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        phase = rad.reshape(f0_frames.shape[0], -1, 1)%1
        return phase
    
    def forward(self, noise_mag, uv_mask, f0_frames):
        '''
            noise_mag: B x n_noise x T
            uv_mask: B x 1 x T
            f0_frames: B x 1 x T
            return: B x T*hop_size
        '''
        B, _, T = noise_mag.shape
        noise_mag_upsamp = upsample(noise_mag, self.hop_size) # b x n_noise x T*hop_size
        uv_mask_upsamp = upsample(uv_mask.unsqueeze(1), self.hop_size) # b x 1 x T*hop_size
        noiseran = replicate(self.noiseran, T*self.hop_size, batch=B) # b x n_noise x T*hop_size

        if self.triangle_ReLU:
            x = self.fast_phase_gen(f0_frames.transpose(1,2)).transpose(1,2) # b x 1 x T*hop_size

            triangle_mask = self.Triangle_ReLU(x, self.triangle_ReLU_up, self.triangle_ReLU_down)
            triangle_mask = triangle_mask * uv_mask_upsamp + (1-uv_mask_upsamp)
            noise_ = noiseran * triangle_mask # b x n_noise x T*hop_size
        else:
            noiseop = replicate(self.noiseop, T*self.hop_size, batch=B)
            noise_ = (noiseop * uv_mask_upsamp + noiseran * (1 - uv_mask_upsamp)) # b x  n_noise x T*hop_size

        noise_ = noiseran * triangle_mask # b x n_noise x T*hop_size
        noise = noise_ * noise_mag_upsamp # b x n_noise x T*hop_size
        noise = noise.sum(dim=1) # b x T*hop_size
        return noise