import numpy as np
import torch
import onnxruntime
import yaml


from tqdm import tqdm
from logger.utils import DotDict
import soundfile as sf
import click
from pathlib import Path
from preprocess import Preprocessor
from ddsp.vocoder import load_model

def infer_onnx(
        model : torch.nn.Module, 
        input : Path, 
        output: Path, 
        args  : DotDict, 
        key   : float, 
        device: str, 
        sample_rate: int
):
    '''
    Args:
        input : audio file path
        output: output audio file path
        key   : the key change in semitones   
    '''
    # Process single file
    print(f"Processing file: {input}")
    audio, sr = sf.read(str(input))

    assert sr == sample_rate, f"\
                            Sample rate of input file {sr} does not match \
                            model sample rate {sample_rate}"
    
    # preprocess
    preprocessor = Preprocessor(args, device)
    mel, f0, uv=preprocessor.mel_f0_uv_process(torch.from_numpy(audio).float())

    print(f"Input shape: {mel.shape}, F0 shape: {f0.shape}, UV shape: {uv.shape}")
    # np.save(output.with_suffix('.npy'), mel)
    

    # forward and save the output
    '''with torch.no_grad():
        if output_f0 is None:
            signal, _, (s_h, s_n), (sin_mag, sin_phase) = model(torch.tensor(mel).float().unsqueeze(0).to(device), torch.tensor(f0).unsqueeze(0).unsqueeze(-1).to(device))
        else:
            signal, _, (s_h, s_n) = model(torch.tensor(mel).float().unsqueeze(0).to(device), torch.tensor(f0).unsqueeze(0).unsqueeze(-1).to(device))
        signal = signal.squeeze().cpu().numpy()
        s_h = s_h.squeeze().cpu().numpy()
        s_n = s_n.squeeze().cpu().numpy()
        sf.write(str(output), signal, args.data.sampling_rate,subtype='FLOAT') 
        sf.write(str(output.with_suffix('.harmonic.wav')), s_h, args.data.sampling_rate,subtype='FLOAT') 
        sf.write(str(output.with_suffix('.noise.wav')), s_n, args.data.sampling_rate,subtype='FLOAT') '''
    #onnx inference
    input_name1 = model.get_inputs()[0].name
    input_name2 = model.get_inputs()[1].name
    output_name = model.get_outputs()[0].name
    input_shape1 = model.get_inputs()[0].shape
    input_shape2 = model.get_inputs()[1].shape
    output_shape = model.get_outputs()[0].shape
    print(f"Input name1: {input_name1}, Input name2: {input_name2}, Output name: {output_name}")
    print(f"Input shape1: {input_shape1}, Input shape2: {input_shape2}, Output shape: {output_shape}")
    # 把mel从[128,n]变成[1,128,n]
    mel = np.expand_dims(mel, axis=0)
    # 把mel从[1,128,n]变成[1,n,128]
    mel = np.transpose(mel, (0, 2, 1))

    f0 = np.expand_dims(f0, axis=0)
    ort_inputs = {input_name1: np.array(mel, dtype=np.float32), input_name2: np.array(f0, dtype=np.float32)}    

    ort_outs = model.run(None, ort_inputs)
    signal = ort_outs[0]
    #把signal从[1,n]变成[n]
    signal = signal.squeeze()
    sf.write(str(output), signal, args.data.sampling_rate,subtype='FLOAT') 


@click.command()
@click.option(
    '--model_path', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=Path, resolve_path=True
    ),
    required=True, metavar='CONFIG_FILE',
    help='The path to the model.'
)
@click.option(
    '--input', type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True,
        path_type=Path, resolve_path=True
    ),
    required=True, 
    help='The path to the WAV file or directory containing WAV files.'
)
@click.option(
    '--output', type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True,
        path_type=Path, resolve_path=True
    ),
    required=True, 
    help='The path to the output directory.'
)
@click.option(
    '--key', type=int, default=0,
    help='key changed (number of semitones)'
)

def main(model_path, input, output, key):

    # cpu inference is fast enough!
    device = 'cpu' 
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #model, args = load_model(model_path, device=device)
    model = onnxruntime.InferenceSession(str(model_path))
    args = DotDict(yaml.load(open("E:/pc-ddsp5.29/configs/SinStack.yaml", 'r'), Loader=yaml.FullLoader))
    print(f"Model loaded: {model_path}")

    if input.is_file():
        infer_onnx(model, input, output / input.name, args, key, device, args.data.sampling_rate)
    elif input.is_dir():
        assert output.is_dir(),\
              "If input is a directory, output must be a directory as well."
        for file in tqdm(input.glob('*.wav')):
            infer_onnx(  
                model,  
                file,  
                output / file.name,  
                args,  
                key,  
                device,  
                args.data.sampling_rate  
            )
if __name__ == '__main__':
    main()