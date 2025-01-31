import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict
import click


from logger import utils
from data_loaders import get_data_loaders
from ddsp.vocoder import SinStack
from ddsp.loss import HybridLoss
from logger.utils import DotDict, load_model
from logger.saver import Saver

class ModelTrainer:
    def __init__(self, config: DotDict, device: str):
        self.args = config
        self.device = device

        self.load_model = load_model

        self.model = SinStack(
            args=config,
            device=device
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters())

        self.initial_global_step, self.model, self.optimizer \
            = self.load_model(
                Path(config.env.expdir), 
                self.model, 
                self.optimizer
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config.train.lr
            param_group['weight_decay'] = config.train.weight_decay

        self.loss_func = HybridLoss(
            config.data.hop_size, 
            config.loss.fft_min, 
            config.loss.fft_max, 
            config.loss.n_scale, 
            config.loss.lambda_uv, 
            config.loss.lambda_ampl, 
            config.loss.lambda_phase, 
            device
        ).to(self.device)

        # device
        if device == 'cuda':
            torch.cuda.set_device(config.env.gpu_id)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        self.loader_train, self.loader_valid \
            = get_data_loaders(self.args, whole_audio=False)

    def train(self):
        saver = Saver(
            self.args,
            initial_global_step=self.initial_global_step
        )
        print(f' [*] experiment dir: {self.args.env.expdir}')

        params_count = utils.get_network_paras_amount({'model': self.model})
        saver.log_info('--- model size ---')
        saver.log_info(params_count)

        best_loss = np.inf
        num_batches = len(self.loader_train)
        self.model.train()
        saver.log_info('======= start training =======')

        for epoch in range(self.args.train.epochs):
            for batch_idx, data in enumerate(self.loader_train):
                saver.global_step_increment()

                self.train_process_batch(
                    saver, 
                    data, 
                    batch_idx, 
                    num_batches, 
                    epoch, 
                    best_loss
                )

    def train_process_batch(
        self, 
        saver: Any, 
        data: Dict[str, Any], 
        batch_idx: int, 
        num_batches: int, 
        epoch: int, 
        best_loss: float
    ):
        self.optimizer.zero_grad()

        # unpack data
        self.move_data_to_device(data)

        # forward
        signal, _, (s_h, s_n), (pre_ampl, pre_phase) = self.model(
            data['mel'], data['f0'], inference=False
        )
        # loss
        detach_uv = saver.global_step < self.args.loss.detach_uv_step
        loss, (loss_rss, loss_uv, loss_ampl, loss_phase) = self.loss_func(
            signal, 
            s_h, 
            pre_ampl, 
            pre_phase, 
            data['audio'], 
            data['uv'], 
            data['ampl'],
            data['phase'],
            detach_uv=detach_uv, 
            uv_tolerance=self.args.loss.uv_tolerance
        )

        # handle nan loss and back propagate
        if torch.isnan(loss):
            raise ValueError(' [x] nan loss ')
        else:
            loss.backward()
            self.optimizer.step()

        self.log_training_progress(
            saver, 
            loss, 
            loss_rss, 
            loss_uv, 
            loss_ampl, 
            loss_phase, 
            batch_idx, 
            num_batches, 
            epoch
        )

        self.validate_and_save(saver, best_loss)

    def move_data_to_device(self, data: Dict[str, Any]):
        for k in data.keys():
            if k != 'name':
                data[k] = data[k].to(self.args.device)
    
    def log_training_progress(
        self, 
        saver: Any, 
        loss: torch.Tensor, 
        loss_rss: torch.Tensor, 
        loss_uv: torch.Tensor, 
        loss_ampl: torch.Tensor, 
        loss_phase: torch.Tensor, 
        batch_idx: int, 
        num_batches: int, 
        epoch: int
    ):
        if saver.global_step % self.args.train.interval_log == 0:
            saver.log_info(
                'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | loss: {:.3f} | rss: {:.3f}| uv: {:.3f} | ampl: {:.3f} | phase: {:.3f} | time: {} | step: {}'.format(
                    epoch,
                    batch_idx,
                    num_batches,
                    self.args.env.expdir,
                    self.args.train.interval_log/saver.get_interval_time(),
                    loss.item(),
                    loss_rss.item(),
                    loss_uv.item(),
                    loss_ampl.item(),
                    loss_phase.item(),
                    saver.get_total_time(),
                    saver.global_step
                )
            )
            
            saver.log_value({
                'train/loss': loss.item(),
                'train/rss': loss_rss.item(),
                'train/uv': loss_uv.item(),
                'train/ampl': loss_ampl.item(),
                'train/phase': loss_phase.item()
            })

    def validate_and_save(self, saver: Any, best_loss: float):
        if saver.global_step % self.args.train.interval_val == 0:
            saver.save_model(self.model, self.optimizer, postfix=f'{saver.global_step}')
            test_loss, test_loss_rss, test_loss_uv, loss_ampl, loss_phase = self.test(self.args, self.loss_func, self.loader_valid, saver)
            saver.log_info(self.get_validation_message(test_loss, test_loss_rss, test_loss_uv, loss_ampl, loss_phase))
            saver.log_value({
                'validation/loss': test_loss,
                'validation/rss': test_loss_rss,
                'validation/uv': test_loss_uv
            })
            #self.update_best_model(saver, best_loss, test_loss)

    def get_validation_message(self, test_loss: float, test_loss_rss: float, test_loss_uv: float, test_loss_ampl: float, test_loss_phase: float) -> str:
        return ' --- <validation> --- \nloss: {:.3f} | rss: {:.3f} | uv: {:.3f}| ampl: {:.3f}| phase: {:.3f}. '.format(test_loss, test_loss_rss, test_loss_uv, test_loss_ampl, test_loss_phase)

    def update_best_model(self, saver: Any, best_loss: float, test_loss: float):
        if test_loss < best_loss:
            saver.log_info(' [V] best model updated.')
            saver.save_model(self.model, self.optimizer, postfix='best')
            best_loss = test_loss

    def test(self, args, loss_func, loader_test, saver):
        print(' [*] testing...')
        self.model.eval()

        # losses
        test_loss = 0.
        test_loss_rss = 0.
        test_loss_uv = 0.
        test_loss_ampl = 0.
        test_loss_phase = 0.
        
        # intialization
        num_batches = len(loader_test)
        rtf_all = []

        # run
        with torch.no_grad():
            for bidx, data in enumerate(loader_test):

                loss, loss_rss, loss_uv, loss_ampl, loss_phase = self.test_process_bath(
                    args, loss_func, saver, num_batches, rtf_all, bidx, data
                )
                test_loss += loss.item()
                test_loss_rss += loss_rss.item()
                test_loss_uv += loss_uv.item()
                test_loss_ampl += loss_ampl.item()
                test_loss_phase += loss_phase.item()

                
        # report
        test_loss /= num_batches
        test_loss_rss /= num_batches
        test_loss_uv /= num_batches
        test_loss_ampl /= num_batches
        test_loss_phase /= num_batches
        
        # check
        print(' [test_loss] test_loss:', test_loss)
        print(' [test_loss_rss] test_loss_rss:', test_loss_rss)
        print(' [test_loss_uv] test_loss_uv:', test_loss_uv)
        print(' [test_loss_ampl] test_loss_ampl:', test_loss_ampl)
        print(' [test_loss_phase] test_loss_phase:', test_loss_phase)
        print(' Real Time Factor', np.mean(rtf_all))
        return test_loss, test_loss_rss, test_loss_uv, test_loss_ampl, test_loss_phase

    def test_process_bath(
            self, 
            args: DotDict, 
            loss_func: HybridLoss, 
            saver, 
            num_batches, 
            rtf_all, 
            bidx, 
            data,
    ):
        fn = data['name'][0]
        print('--------')
        print('{}/{} - {}'.format(bidx, num_batches, fn))

        # unpack data
        self.move_data_to_device(data)
        print('>>', data['name'][0])

        # forward
        st_time = time.time()
        signal, _, (s_h, s_n), (pre_ampl, pre_phase) = self.model(data['mel'], data['f0'], infer=False)
        ed_time = time.time()

        # crop. 因为test的时候，audio的长度不一定等于block_size的整数倍。
        signal = self.crop_audio(data, signal)
        
        # RTF
        self.compote_RTF(args, rtf_all, data, ed_time - st_time)

        # log
        saver.log_audio({fn+'/gt.wav': data['audio'], fn+'/pred.wav': signal})
            
        # loss
        loss, (loss_rss, loss_uv, loss_ampl, loss_phase) = loss_func(
            signal, 
            s_h, 
            pre_ampl, 
            pre_phase, 
            data['audio'], 
            data['uv'], 
            data['ampl'], 
            data['phase'], 
            detach_uv=True
        )

        return loss, loss_rss, loss_uv, loss_ampl, loss_phase


    def compote_RTF(self, args, rtf_all, data, run_time):
        song_time = data['audio'].shape[-1] / args.data.sampling_rate
        rtf = run_time / song_time
        print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
        rtf_all.append(rtf)

    def crop_audio(self, data, signal):
        min_len = np.min([signal.shape[1], data['audio'].shape[1]])
        signal        = signal[:,:min_len]
        data['audio'] = data['audio'][:,:min_len]
        return signal

@click.command()
@click.option(
    '--config', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=Path, resolve_path=True
    ),
    required=True, metavar='CONFIG_FILE',
    help='The path to the config file.'
)
def main(config):
    print(' > starting training...')

    # load config
    args = utils.load_config(config)
    print(' > config:', config)
    print(' >    exp:', args.env.expdir)

    trainer = ModelTrainer(args, args.device)
    trainer.train()


if __name__ == '__main__':
    main()
