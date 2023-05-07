import warnings
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from pyemd import emd, emd_with_flow
from collections import defaultdict

from util import AverageMeter


# The following is adapted from the Jupyter notebook '2. Deep-STPP Visualization'
# maybe the inference limit should be outside of (training) config

class MeshInfo:
    def __init__(self, mesh_centre, mesh_area: int, mesh_region=[]):
        self.mesh_centre = mesh_centre
        self.mesh_area = mesh_area
        self.num_mesh = len(mesh_centre)
        self.mesh_region = mesh_region
    def update_mesh_region(self, mesh_region):
        self.mesh_region = mesh_region


# biases:
# start_date: 
class DataInfo:
    def __init__(self, biases, scales, start_date) -> None:
        self.biases = biases
        self.scales = scales
        self.start_date = start_date

class Params:
    def __init__(self):
        self.w_i = 0
        self.b_i = 0
        self.inv_var = 0
    def update(self, w_i, b_i, inv_var):
        self.w_i = w_i
        self.b_i = b_i
        self.inv_var = inv_var

class ModelPredictor:
    def __init__(self, model, config, device, data_info, mesh_info: MeshInfo):
        
        self.model = model
        self.background = model.background.cpu().detach()
        self.num_bg = self.background.shape[0]
        
        self.config = config
        self.device = device

        self.data_info = data_info
        self.mesh_info = mesh_info

        self.params = Params()
        
    # get_model_output: get the output from the model based on the given input st_x
    #   INPUT:  st_x: input data, tensor of dimension [seq_len, num_feature]
    #   OUTPUT: params: a tuple containing (w_i, b_i, variance)
    def _update_params(self, st_x):
        _, w_i, b_i, inv_var = self.model(st_x.unsqueeze(dim=0).to(self.device))
        w_i  = w_i.cpu().detach()
        b_i  = b_i.cpu().detach()
        inv_var = inv_var.cpu().detach()
        self.params.update(w_i, b_i, inv_var)

    # s_kernel_mesh: calculate the spatial kernel based on the input
    #   INPUT:  st_x: input data, Tensor of dimension (seq_len, num_feature)
    #   OUTPUT: s_kernel: spatial kernel, Tensor of dimension (num_mesh, 1)
    def _s_kernel_mesh(self, st_x):

        # get the parameters    
        inv_var = self.params.inv_var.repeat(self.mesh_info.num_mesh, 1, 1)

        # scale the mesh_centre
        bias_xy = torch.tensor(self.data_info.biases[0:2])
        scale_xy = torch.tensor(self.data_info.scales[0:2])
        s_grids = (self.mesh_info.mesh_centre - bias_xy) / scale_xy
        
        st_x = torch.cat((st_x[..., :-1], self.background), 0).unsqueeze(0).repeat(self.mesh_info.num_mesh, 1, 1)
        s_diff = s_grids.unsqueeze(1) - st_x

        s_kernel = torch.sum(s_diff * inv_var * s_diff, -1)
        s_kernel = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5*s_kernel)/(2*np.pi)  

        return s_kernel

    def _t_integral_mesh(self, st_x_cum, time_period):

        b_i = self.params.b_i

        time_from, time_to = time_period
        bias_t = self.data_info.biases[-1]
        scale_t = self.data_info.scales[-1]
        
        tn_ti = torch.cat((st_x_cum[-1,-1]-st_x_cum[:,2], torch.zeros(self.num_bg)), 0)
        tp_ti_scaled = torch.sub(torch.add(tn_ti, time_from), bias_t) /scale_t
        tq_ti_scaled = torch.sub(torch.add(tn_ti, time_to), bias_t) /scale_t

        t_integral = 1/b_i * (torch.exp(-b_i*tp_ti_scaled) - torch.exp(-b_i*tq_ti_scaled))

        return t_integral


    def calc_expected_event_mesh(self, s_kernel, st_x_cum, time_period):

        s_integral = s_kernel * self.mesh_info.mesh_area / np.product(self.data_info.scales[:2])
        t_integral = self._t_integral_mesh(st_x_cum, time_period)

        expected_event_count = torch.sum(s_integral * t_integral * self.params.w_i, -1).numpy()
        
        return expected_event_count


    # INPUT: 
    #       test_loader (DataLoader): An iterator of test sequences.
    #       time_duration (int): How many days ahead we would like to predict (for integrating t)?
    #       daily_freq (int): How often (relative to days) do we want to generate predictions?
    #           (e.g. 2 = make two predictions in one day, or every 12 hours.)
    # OUTPUT: 
    #       df_count: A tensor of dimension (num_rep, num_mesh, num_interval) representing the event count.

    def pred_mesh_count(self, test_loader, time_period=1, daily_freq=1):

        # Perform evaluation using the input data and internal state
        # Create a DataFrame with the predicted results

        ### concatenate all sequences in test_loader
        st_x_s = []
        st_y_s = []
        st_x_org_s = []
        st_y_org_s = []
        seq_ind_s = []

        for input_batch in test_loader:

            st_x, st_y, st_x_org, st_y_org, seq_ind = input_batch

            st_x_s += st_x
            st_y_s += st_y
            st_x_org_s += st_x_org
            st_y_org_s += st_y_org
            seq_ind_s += seq_ind[1]

        rep = -1               # to track the number of reps of the data
        num_seq = len(st_x_s)  
        mesh_count = []
            
        for curr_ind, (st_x, st_y, st_x_org, st_y_org, seq_ind) in enumerate(tqdm(zip(st_x_s, st_y_s, st_x_org_s, st_y_org_s, seq_ind_s), total=num_seq)):          
            
            if curr_ind == 0:
                seq_ind_prev = np.inf
                
            else:
                seq_ind_prev = seq_ind_s[curr_ind-1]

            if seq_ind_s[curr_ind] < seq_ind_prev:
                rep += 1
                mesh_count.append([])
                cutoff_time = torch.ceil(st_x_org[-1,-1]).item()
                if curr_ind == 0:
                    starting_time = cutoff_time
                    print(f'Prediction starts from day {starting_time} with interval {1/daily_freq}.')
            
            last_t_x = st_x_org[-1,-1]
            last_t_y = st_y_org[0][-1]
                    
            if (last_t_x < cutoff_time) and (last_t_y >= cutoff_time):
                
                self._update_params(st_x)
                s_kernel = self._s_kernel_mesh(st_x)
                
                while last_t_y >= cutoff_time:
                    time_period = (cutoff_time - last_t_x, cutoff_time - last_t_x + 1 / daily_freq)
                    expected_event_count = self.calc_expected_event_mesh(s_kernel, st_x_org, time_period)
                    mesh_count[rep].append(expected_event_count)
                    cutoff_time += 1 / daily_freq
        
        print(f'Prediction ends on {cutoff_time}. # of variant sequence: {rep+1}.')

        mesh_count = np.array(mesh_count)
        mesh_count_avg = np.mean(mesh_count, axis=0)
        mesh_count_avg = pd.DataFrame(mesh_count_avg.transpose())

        # drop columns in between each day
        num_days = int(np.floor(cutoff_time - starting_time))
        index_daily = list(range(0, num_days, daily_freq))
        mesh_count_avg = mesh_count_avg[index_daily]
        start_date_pd = pd.Timestamp(self.data_info.start_date) + pd.Timedelta(starting_time,'D')

        index_new = pd.period_range(start_date_pd, periods=num_days, freq='D')
        mesh_count_avg.columns = index_new

        return mesh_count, mesh_count_avg
    
    def pred_agg_count(self, mesh_count_avg):

        if len(self.mesh_info.mesh_region) == 0:
            raise ValueError("No mapping between mesh and regions found. Please update the mapping first.")

        mesh_count_avg_copy = mesh_count_avg.copy(deep=True)
        mesh_count_avg_copy['region_id'] = self.mesh_info.mesh_region
        agg_count = mesh_count_avg_copy.groupby(['region_id']).sum()

        # dtype might be float due to NaNs. change the dtype to int.
        if np.dtype(agg_count.index) == 'float': 
            agg_count.index = agg_count.index.astype(int)
            
        return agg_count
            
    
class ModelEvaluator():
    def __init__(self, aggregator, y_pred) -> None:
        
        test_date_start = min(y_pred.columns)
        test_date_end = max(y_pred.columns)
        y_true_period = aggregator.panel['period']

        daily_data = aggregator.panel[(y_true_period>=test_date_start)*(y_true_period<=test_date_end)]

        y_true = daily_data[['event_count','cell_id','period']].pivot(
            index='cell_id',columns='period',values='event_count'
            ).sort_index()
        
        # add lines of 0 for (tiny) admin regions that contains no mesh centres
        y_pred = y_pred.align(y_true, axis=0, fill_value=0)[0].sort_index()
        
        if y_true.shape != y_pred.shape:
            print(y_true.shape, y_pred.shape)
            raise ValueError('Dimensions do not match between groundtruth and predictions.')
        
        self.aggregator = aggregator
        self.y_true = y_true
        self.y_pred = y_pred

    def calc_pseudoEMD(self, 
                       alpha=1,
                       dist_metric='euclidean', 
                       unit_scale=True, 
                       **kwargs):

        dist_mat = self.aggregator.get_dist_matrix(metric=dist_metric)  
        if unit_scale:
            # Optionally scale by maximum distance
            dist_mat = dist_mat / dist_mat.max()
        max_dist = dist_mat.max()
        penalty = alpha * max_dist

        y_pred_np = self.y_pred.to_numpy()
        y_true_np = self.y_true.to_numpy()
        emd_meter = AverageMeter()

        for col_ind in range(y_true_np.shape[1]):
            y_pred_col = y_pred_np[:,col_ind]
            y_true_col = y_true_np[:,col_ind]

            mismatch = emd(
                y_pred_col, 
                y_true_col, 
                dist_mat, 
                extra_mass_penalty=penalty
            )
            emd_meter.update(mismatch)

        return emd_meter
        
    def calc_score(self, metrics, rolling_period=1, **kwargs):
        y_true = self.y_true.rolling(rolling_period, min_periods=1, axis=1).mean()
        y_pred = self.y_pred.rolling(rolling_period, min_periods=1, axis=1).mean()
        return metrics(y_true, y_pred, **kwargs)
