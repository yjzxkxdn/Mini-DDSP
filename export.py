import argparse
import os.path

import torch

from ddsp.vocoder import load_model


class DDSPWrapper(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.model = module
        self.to(device)

    def forward(self, mel, f0):
        mel = mel.transpose(1, 2)
        f0 = f0[..., None]
        signal, _, (harmonic, noise), (sin_mag, sin_phase) = self.model(mel, f0)
        print(f' [Output] signal: {signal.shape}, harmonic: {harmonic.shape}, noise: {noise.shape}, sin_mag: {sin_mag.shape}, sin_phase: {sin_phase.shape}')
        return signal

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description='Export model to standalone PyTorch traced module or ONNX format'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='path to model file'
    )
    parser.add_argument(
        '--traced',
        required=False,
        action='store_true',
        help='export to traced module format'
    )
    parser.add_argument(
        '--onnx',
        required=False,
        action='store_true',
        help='export to ONNX format'
    )
    cmd = parser.parse_args(args=args, namespace=namespace)
    if not cmd.traced and not cmd.onnx:
        parser.error('either --traced or --onnx should be specified.')
    return cmd


def main():
    device = 'cpu'
    # parse commands
    cmd = parse_args()

    # load model
    model, args = load_model(cmd.model_path, device=device)
    model = DDSPWrapper(model, device)

    # extract model dirname and filename
    directory = os.path.dirname(os.path.abspath(cmd.model_path))
    name = os.path.basename(cmd.model_path).rsplit('.', maxsplit=1)[0]

    # load input
    n_mel_channels = args.data.n_mels
    n_frames = 10
    mel = torch.randn((1, n_frames, n_mel_channels), dtype=torch.float32, device=device)
    f0 = torch.FloatTensor([[440.] * n_frames]).to(device)
    print(f' [Input] mel: {mel.shape}, f0: {f0.shape}')
    
    # export model
    with torch.no_grad():
        if cmd.traced:
            torch_version = torch.version.__version__.rsplit('+', maxsplit=1)[0]
            export_path = os.path.join(directory, f'{name}-traced-torch{torch_version}.jit')
            print(f' [Tracing] {cmd.model_path} => {export_path}')
            model = torch.jit.trace(
                model,
                (
                    mel,
                    f0
                ),
                check_trace=False
            )
            torch.jit.save(model, export_path)

        elif cmd.onnx:
            # Prepare the export path for ONNX format
            onnx_version = "unknown"
            try:
                import onnx
                onnx_version = onnx.__version__
            except ImportError:
                print("Warning: ONNX package is not installed. Please install it to enable ONNX export.")
                return
            
            export_path = os.path.join(directory, f'{name}-torch{torch.version.__version__[:5]}-onnx{onnx_version}.onnx')
            print(f' [Exporting] {cmd.model_path} => {export_path}')

            # Export the model to ONNX
            torch.onnx.export(
                model, 
                (mel, f0), 
                export_path, 
                export_params=True, 
                opset_version=15,
                input_names=['mel', 'f0'], 
                output_names=['output'], 
                dynamic_axes={'mel': {1: 'n_frames'}, 'f0': {1: 'n_frames'},
                              'output': {1: 'n_samples'}} 
            )

            print(f"Model has been successfully exported to {export_path}")
def simplify_onnx_model():

    import onnx
    from onnxsim import simplify

    model = onnx.load(r"E:\pc-ddsp5.29\exp\qixuan8\model_19000-torch2.1.0-onnx1.16.2.onnx")

    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, 'output_model_simplified.onnx')

if __name__ == '__main__':
    main()
    #simplify_onnx_model()