import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from app.gleam_ml.lib import utils
from app.gleam_ml.dcrnn_model import DCRNNModel
from app.gleam_ml.loss import mae_loss
from app.gleam_ml.loss import mae_metric
from app.gleam_ml.loss import rmse_metric
from app.gleam_ml.loss import kld_gaussian_loss
# from model.pytorch.loss import maxentropy
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STNPSupervisor:
    def __init__(self, random_seed, iteration, max_itr, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.iteration = iteration
        self.max_itr = max_itr

        # logging.
        self._log_dir = self._get_log_dir(kwargs, self.random_seed, self.iteration)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.input0_dim = int(self._model_kwargs.get('input0_dim', 24))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda(device) if torch.cuda.is_available() else dcrnn_model
        self.z_mean_all=None
        self.z_var_temp_all=None
        self.num_batches = None #int(0)
        self.batch_size = int(self._data_kwargs.get('batch_size'))
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs, random_seed, iteration):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'

            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s_%d_%d/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'), random_seed, iteration)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def load_model(self):
        assert os.path.exists('app/gleam_ml/z_epo101.tar'), 'Z at epoch %d not found' % self._epoch_num
        checkpoint1 = torch.load('app/gleam_ml/z_epo101.tar', map_location='cpu')
        self.z_mean_all = checkpoint1[0].to(device)
        self.z_var_temp_all = checkpoint1[1].to(device)
        assert os.path.exists('app/gleam_ml/model_epo101.tar'), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('app/gleam_ml/model_epo101.tar', map_location='cpu')

        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.dcrnn_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.dcrnn_model.load_state_dict(model_dict)
 
        self._logger.info("Loaded model at {}".format(self._epoch_num))


    def _prepare_data(self, x, x0):
        x, x0 = self._get_x_y(x, x0)
        x, x0 = self._get_x_y_in_correct_dims(x, x0)
        return x.to(device), x0.to(device)

    def _get_x_y(self, x, x0):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param x0: shape (batch_size, input_dim_startingpoint)
        :param y: shape (batch_size, seg_len, output_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 x0 shape (batch_size, input_dim_startingpoint)
                 y shape (seq_len, batch_size, output_dim)
        """
        x = torch.from_numpy(x).float()
        x0 = torch.from_numpy(x0).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("X0: {}".format(x0.size()))
        x = x.permute(1, 0, 2, 3)
        return x, x0

    def _get_x_y_in_correct_dims(self, x, x0):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param x0: shape (batch_size, input_dim_startingpoint)
        :param y: shape (horizon, batch_size, output_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 x0: shape (batch_size, input_dim_startingpoint)
                 y: shape (seq_len, batch_size, output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        return x, x0
    
    def predict(self, x, x0):
        inputer = len(x)
        z_var_all = 0.1 + 0.9 * torch.sigmoid(self.z_var_temp_all)
        zs = self.dcrnn_model.sample_z(self.z_mean_all, z_var_all, inputer)
        outputs_hidden = self.dcrnn_model.dcrnn_to_hidden(x)
        output = self.dcrnn_model.decoder(x0, outputs_hidden, zs)
        return output