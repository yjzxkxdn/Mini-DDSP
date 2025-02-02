import os
import yaml
import json
import torch
from pathlib import Path
from typing  import Tuple, Any, Dict, TypeVar, Union

def traverse_dir(
        root_dir:     Path,
        extension:    str ,
        amount:       int  = None ,
        str_include:  str  = None ,
        str_exclude:  str  = None ,
        is_pure:      bool = False,
        is_sort:      bool = False,
        is_ext:       bool = True
) -> list:
    """
    遍历目录，返回指定后缀的文件列表。

    Args:
        root_dir:    根目录
        extension:   文件后缀
        amount:      最大返回数量 (None 表示返回所有文件)
        str_include: 包含字符串
        str_exclude: 排除字符串
        is_pure:     是否返回相对路径
        is_sort:     是否按文件名排序
        is_ext:      是否包含后缀

    Returns:
        文件列表
    """
    root_dir = Path(root_dir).resolve()  # Ensure the path is absolute
    file_list = []

    for path in root_dir.rglob(f"*{'.' + extension if extension is not None else ''}"):
        path_str = str(path)
        relative_path = path_str[len(str(root_dir)) + 1:] if is_pure else path_str

        if str_include and str_include not in relative_path:
            continue
        if str_exclude and str_exclude in relative_path:
            continue

        if not is_ext:
            relative_path = '.'.join(relative_path.split('.')[:-1])

        file_list.append(relative_path)

        if amount is not None and len(file_list) >= amount:
            break

    if is_sort:
        file_list.sort()

    return file_list

T = TypeVar('T', bound='DotDict')

class DotDict(Dict[str, Any]):
    """
    DotDict 可以让字典用点符号访问。类似于访问对象。
    和pc-ddsp中的DotDict基本一致，并且嵌套字典也可以。
    
    例子：
        config = DotDict({'key': {'nested_key': 'value'}})
        print(config.key.nested_key)  
        输出: 'value'
    """
    def __init__(self, *args: Union[Dict[str, Any], 'DotDict'], **kwargs: Any) -> None:
        super().__init__()
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    self[key] = self._convert(value)
            elif isinstance(arg, DotDict):
                self.update(arg)
        for key, value in kwargs.items():
            self[key] = self._convert(value)

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, DotDict):
            return DotDict(value)
        return value

    def __getattr__(self: T, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                self[name] = value = DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._convert(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")


def get_network_paras_amount(model_dict: Dict[str, torch.nn.Module]) -> Dict[str, Tuple[int, int]]:
    """
    计算模型参数数量。和pc-ddsp中的函数基本一致，多了参数总量和可训练参数总量的输出。

    Args:
        model_dict: 模型字典.

    Returns:
        (total_params, trainable_params).
    """
    info = {}
    for model_name, model in model_dict.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info[model_name] = (total_params, trainable_params)
    
    return info


def load_config(path_config: Path) -> 'DotDict':
    """
    修改自 pc-ddsp 中的 load_config 函数，添加了检查。
    加载并解析给定路径的 YAML 配置文件，返回一个 DotDict 对象，
    允许使用点符号访问配置项。

    Args:
        path_config: 配置文件的路径

    Return:
        包含配置数据的 DotDict 对象
    """
    try:
        with open(path_config, "r") as config_file:
            args = yaml.safe_load(config_file)
            return DotDict(args)
    except FileNotFoundError:
        raise ValueError(f"配置文件未找到: {path_config}")
    except yaml.YAMLError:
        raise ValueError(f"YAML 文件格式错误: {path_config}")
    
def validate_config(config: DotDict) -> None:
    """
    校验配置文件的正确性，抛出异常如果发现任何问题。

    Args:
        config: 包含配置数据的 DotDict 对象
    """
    # 数据部分的校验
    assert config.data.f0_extractor in ['parselmouth', 'dio', 'harvest'], "f0_extractor 必须是 'parselmouth', 'dio', 或 'harvest'"
    assert isinstance(config.data.f0_min, int), "f0_min 必须是整数"
    assert isinstance(config.data.f0_max, int), "f0_max 必须是整数"
    assert config.data.f0_min < config.data.f0_max, "f0_min 必须小于 f0_max"
    assert config.data.sampling_rate > 0, "sampling_rate 必须大于 0"
    assert config.data.n_fft > 0, "n_fft 必须大于 0"
    assert config.data.win_length > 0, "win_length 必须大于 0"
    assert config.data.block_size > 0, "block_size 必须大于 0"
    assert config.data.block_size == config.data.win_length // 2, "block_size 应等于 win_length 的一半"
    assert config.data.n_mels > 0, "n_mels 必须大于 0"
    assert config.data.mel_fmin >= 0, "mel_fmin 必须大于或等于 0"
    assert config.data.mel_fmax > config.data.mel_fmin, "mel_fmax 必须大于 mel_fmin"
    assert config.data.duration > 0, "duration 必须大于 0"
    assert isinstance(config.data.train_path, str), "train_path 必须是字符串"
    assert isinstance(config.data.valid_path, str), "valid_path 必须是字符串"

    # 模型部分的校验
    assert config.model.type in ['CombSub'], "model.type 必须是 'CombSub'"
    assert config.model.win_length > 0, "model.win_length 必须大于 0"
    assert isinstance(config.model.use_mean_filter, bool), "use_mean_filter 必须是布尔值"
    assert config.model.n_mag_harmonic > 0, "n_mag_harmonic 必须大于 0"
    assert config.model.n_mag_noise > 0, "n_mag_noise 必须大于 0"

    # 损失函数部分的校验
    assert config.loss.fft_min > 0, "fft_min 必须大于 0"
    assert config.loss.fft_max > config.loss.fft_min, "fft_max 必须大于 fft_min"
    assert config.loss.n_scale > 0, "n_scale 必须大于 0"
    assert config.loss.lambda_uv > 0, "lambda_uv 必须大于 0"
    assert config.loss.uv_tolerance >= 0, "uv_tolerance 必须大于或等于 0"
    assert config.loss.detach_uv_step > 0, "detach_uv_step 必须大于 0"

    # 设备部分的校验
    assert config.device in ['cuda', 'cpu'], "device 必须是 'cuda' 或 'cpu'"

    # 环境部分的校验
    assert isinstance(config.env.expdir, str), "expdir 必须是字符串"
    assert isinstance(config.env.gpu_id, int), "gpu_id 必须是整数"

    # 训练部分的校验
    assert config.train.num_workers >= 0, "num_workers 必须大于或等于 0"
    assert config.train.batch_size > 0, "batch_size 必须大于 0"
    assert config.train.epochs > 0, "epochs 必须大于 0"
    assert config.train.interval_log > 0, "interval_log 必须大于 0"
    assert config.train.interval_val > 0, "interval_val 必须大于 0"
    assert config.train.lr > 0, "lr 必须大于 0"
    assert config.train.weight_decay >= 0, "weight_decay 必须大于或等于 0"



def to_json(path_params: str, path_json: str) -> None:
    # 修改自pc-ddsp，功能不变。
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict: Dict[str, Any] = {k: v.tolist() 
                                      for k, v in params.items()}
                                      
    # 先把字典转成json，这样可以减少文件锁定的时间。
    with open(path_json, 'w') as outfile:
        json_data = json.dumps(raw_state_dict, indent='\t')
        outfile.write(json_data)
           
def load_model(
        expdir: Path, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        name: str = 'model',
        postfix: str = '',
        device: str = 'cpu'
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """
    加载模型和优化器。
    Args:
        expdir:      保存模型的目录
        model:       模型
        optimizer:   优化器
        name:        模型名称
        postfix:     后缀
    Returns:
        global_step: 全局步数
        model:       模型
        optimizer:   优化器
    """
    if postfix:
        postfix = '_' + postfix
    
    path = expdir / (name + postfix)
    path_pt = list(expdir.glob('*.pt'))
    
    global_step = 0
    if path_pt:
        steps = []
        for p in path_pt:
            if p.name.startswith(name + postfix):
                step = p.name[len(name + postfix)+1:].split('.')[0]
                if step == "best":
                    steps = [-1]
                    break
                else:
                    steps.append(int(step))
                  
        maxstep = max(steps or [0])
        
        if maxstep > 0:
            path_pt = path.with_name(f'{path.name}_{maxstep}.pt')
        else:
            path_pt = path.with_name(f'{path.name}_best.pt')
        
        print(' [*] Restoring model from', path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt.get('global_step', 0)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    return global_step, model, optimizer
