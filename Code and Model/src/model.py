import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from tqdm.auto import tqdm

"""
Return a square attention mask to only allow self-attention layers to attend the earlier positions
"""
def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


"""
Injects some information about the relative or absolute position of the tokens in the sequence
ref: https://github.com/harvardnlp/annotated-transformer/
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device("cpu")

    def forward(self, x, t):
        pe = torch.zeros(self.max_len, self.d_model).to(self.device)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(self.device)
        
        t = t.unsqueeze(-1)
        pe = torch.zeros(*t.shape[:2], self.d_model).to(self.device)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        
        x = x + pe[:x.size(0)]
        return self.dropout(x)
    
    
"""
Encode time/space record to variational posterior for location latent
"""
class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(config.emb_dim, config.dropout,
                                              config.seq_len)
        encoder_layers = nn.TransformerEncoderLayer(config.emb_dim, config.num_head,
                                                    config.hid_dim, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.nlayers)
        self.seq_len = config.seq_len
        self.ninp = config.emb_dim
        self.encoder = nn.Linear(3 + config.num_marks, config.emb_dim, bias=False)
        self.decoder = nn.Linear(config.emb_dim, config.z_dim * 2)
        self.init_weights()
        self.device = device
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, x_mask=None):
        x = x.transpose(1, 0)  # Convert to seq-len-first
        if x_mask is None:
            x_mask = subsequent_mask(len(x)).to(self.device)
        t = torch.cumsum(x[..., -1], 0)
        x = self.encoder(x) * math.sqrt(self.ninp)
        x = self.pos_encoder(x, t)
        
        output = self.transformer_encoder(x, x_mask)
        output = self.decoder(output)
        
        output = output[-1]  # get last output only; [batch, z_dim * 2]
        m, v_ = torch.split(output, output.size(-1) // 2, dim=-1)
        v = F.softplus(v_) + 1e-5
        return m, v
    
"""
Decode latent variable to spatiotemporal kernel coefficients
"""
class Decoder(nn.Module):
    def __init__(self, config, out_dim, softplus=False):
        super().__init__()
        self.z_dim = config.z_dim
        self.softplus = softplus
        self.net = nn.Sequential(
            nn.Linear(config.z_dim, config.hid_dim),
            nn.ELU(),
            *[nn.Linear(config.hid_dim, config.hid_dim),
            nn.ELU()] * (config.decoder_n_layer - 1),
            nn.Linear(config.hid_dim, out_dim),
        )

    def decode(self, z):
        output = self.net(z)
        if self.softplus:
            output = F.softplus(output) + 1e-5
        return output
    
    
def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    z = torch.randn_like(m)
    z = z * torch.sqrt(v) + m
    return z


"""
Log likelihood of no events happening from t_n to t
- ∫_{t_n}^t λ(t') dt' 

tn_ti: [batch, seq_len]
t_ti: [batch, seq_len]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: scalar
"""
def ll_no_events(w_i, b_i, tn_ti, t_ti):
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


def log_ft(t_ti, tn_ti, w_i, b_i):
    return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))

"""
Compute spatial/temporal/spatiotemporal intensities

tn_ti: [batch, seq_len]
s_diff: [batch, seq_len, 2]
inv_var = [batch, seq_len, 2]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: λ(t) [batch]
return: f(s|t) [batch] 
return: λ(s,t) [batch]
"""
def t_intensity(w_i, b_i, t_ti):
    v_i = w_i * torch.exp(-b_i * t_ti)
    lamb_t = torch.sum(v_i, -1)
    return lamb_t

def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1) # normalize
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5*g2)/(2*np.pi)
    f_s_cond_t = torch.sum(g2 * v_i, -1)
    return f_s_cond_t

def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)



"""
STPP model with VAE: directly modeling λ(s,t)
"""
class DeepSTPP(nn.Module):
    """
    Updated by Tian You (TY): Modified version of the DeepSTPP model, which
        1) allows the user to provide additional event semantics (='marks'), and
        2) to control the initialisation of background points.
    """
    def __init__(self, config: DeepSTPPConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.device = device
        
        self.num_marks = config.num_marks       # Updated by TY
        self.num_points = config.num_points     # Updated by TY

        # VAE for predicting spatial intensity
        self.enc = Encoder(config, device)
        
        # Make separate decoders for w, b, and s
        output_dim = config.seq_len + config.num_points
        self.w_dec = Decoder(config, output_dim, softplus=True)
        self.b_dec = Decoder(config, output_dim)
        self.s_dec = Decoder(config, output_dim * 2, softplus=True)
        
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        
        # Updated by TY: We allow the user to provide custom initialisations for 
        # background points to ensure that said points are not initialised outside 
        # the spatial domain of interest.
        if isinstance(config.point_inits, str):
            if config.point_inits.lower() == 'uniform':
                inits = torch.rand((self.num_points, 2))
            else:
                raise ValueError(
                    f"'{config.point_inits}' is not a recognised initialisation."
                )
        else:
            inits = torch.from_numpy(config.point_inits)
        self.background = nn.Parameter(inits, requires_grad=True)

        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.to(device)

    def loss(self, st_x, st_y):
        """
        Updated by TY: Added additional marks to the model.
        st_x: [batch, seq_len, 3 + num_marks] (s_1, s_2, mark_1, mark_2, ..., time)
        st_y: [batch, 1, 3 + num_marks]
        """
        batch = st_x.shape[0]
        time_idx = 2 + self.num_marks

        # Note that our loss does not depend on the optional marks,
        # but only on spatio-temporal locations.
        space_hist = st_x[..., :2]
        space_fut = st_y[..., :2]
        time_hist = st_x[..., time_idx]
        time_fut = st_y[..., time_idx]

        background = self.background.unsqueeze(0).repeat(batch, 1, 1)
        s_diff = space_fut - torch.cat((space_hist, background), 1)  # s - s_i
        t_cum = torch.cumsum(time_hist, -1)

        tn_ti = t_cum[..., -1:] - t_cum  # t_n - t_i
        tn_ti = torch.cat((tn_ti, torch.zeros(batch, self.num_points).to(self.device)), -1)
        t_ti = tn_ti + time_fut  # t - t_i

        [qm, qv], w_i, b_i, inv_var = self(st_x)

        # Calculate likelihood
        sll = torch.log(s_intensity(w_i, b_i, t_ti, s_diff, inv_var))
        tll = log_ft(t_ti, tn_ti, w_i, b_i)

        # KL Divergence
        if self.config.sample:
            kl = kl_normal(qm, qv, *self.z_prior).mean()
            nelbo = kl - self.config.beta * (sll.mean() + tll.mean())
        else:
            nelbo = - (sll.mean() + tll.mean())

        return nelbo, sll, tll
   
    
    def forward(self, st_x):        
        # Encode history locations and times
        if self.config.sample:
            qm, qv = self.enc.encode(st_x) # Variational posterior
            z = sample_gaussian(qm, qv)
        else:
            qm, qv = None, None
            z, _ = self.enc.encode(st_x)
        
        w_i = self.w_dec.decode(z)
        b_i = self.decode_b(z)
        s_i = self.s_dec.decode(z) + self.config.s_min
        s_x, s_y = torch.split(s_i, s_i.size(-1) // 2, dim=-1)
        inv_var = torch.stack((1 / s_x, 1 / s_y), -1)

        return [qm, qv], w_i, b_i, inv_var

    def decode_b(self, z):
        '''
        Updated by TY: put decode_b in a seperate function.
        '''
        if self.config.constrain_b == 'tanh':
            b_i = torch.tanh(self.b_dec.decode(z)) * self.config.b_max
        elif self.config.constrain_b == 'sigmoid':
            b_i = torch.sigmoid(self.b_dec.decode(z)) * self.config.b_max
        elif self.config.constrain_b == 'neg-sigmoid':
            b_i = - torch.sigmoid(self.b_dec.decode(z)) * self.config.b_max
        elif self.config.constrain_b == 'softplus':
            b_i = F.softplus(self.b_dec.decode(z))
        elif self.config.constrain_b == 'clamp':
            b_i = torch.clamp(self.b_dec.decode(z), -self.config.b_max, self.config.b_max)
        else:
            b_i = self.b_dec.decode(z)
        return b_i

    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)


"""
Calculate the uniformly samplded spatiotemporal intensity with a given
number of spatiotemporal steps  
"""
def calc_lamb(model,
              test_loader,
              config,
              device,
              *,
              seq_num: int = 0,
              scales=None,
              biases=None,
              x_nstep=101,
              y_nstep=101,
              total_time=None,
              num_frames_per_time_step: int = 2,
              xmax=None,
              xmin=None,
              ymax=None,
              ymin=None,
              verbose=True):
    """
    Modified evaluation of the intensity function, which supports
    selection of a specific sequence number from the test set, while
    using a hard-coded discretisation of time.
    """
    scales = np.ones(3) if scales is None else scales
    biases = np.ones(3) if biases is None else biases

    # Sequence reconstruction:
    #   Recall that the `SlidingWindowWrapper` splits our 'macroscopic'
    #   source sequences into fixed-length 'micro-'sequences, which are
    #   then collected into batches. Crucially, one batch may contain
    #   'micro-'sequences from more than on macroscopic source sequence.
    #   The loop below hence iterates through all test batches to re-construct
    #   the original data.
    st_xs = []
    st_ys = []
    st_x_cums = []
    st_y_cums = []
    for batch in test_loader:
        st_x, st_y, st_x_cum, st_y_cum, (idx, _) = batch
        in_seq = idx == seq_num
        if not torch.any(in_seq):
            continue
        st_xs.append(st_x[in_seq])
        st_ys.append(st_y[in_seq])
        st_x_cums.append(st_x_cum[in_seq])
        st_y_cums.append(st_y_cum[in_seq])

    # Stack the sequence
    st_x = torch.cat(st_xs, 0).cpu()
    st_y = torch.cat(st_ys, 0).cpu()
    st_x_cum = torch.cat(st_x_cums, 0).cpu()
    st_y_cum = torch.cat(st_y_cums, 0).cpu()

    if total_time is None:
        total_time = st_y_cum[-1, -1, -1]

    if verbose:
        print(f'Intensity time range : {total_time}')
    lambs = []

    # Discretize space
    if xmax is None:
        xmax = 1.0
        xmin = 0.0
        ymax = 1.0
        ymin = 0.0
    else:
        xmax = (xmax - biases[0]) / scales[0]
        xmin = (xmin - biases[0]) / scales[0]
        ymax = (ymax - biases[1]) / scales[1]
        ymin = (ymin - biases[1]) / scales[1]

    x_step = (xmax - xmin) / (x_nstep - 1)
    y_step = (ymax - ymin) / (y_nstep - 1)
    x_range = torch.arange(xmin, xmax + 1e-10, x_step)
    y_range = torch.arange(ymin, ymax + 1e-10, y_step) 
    s_grids = torch.stack(torch.meshgrid(x_range, y_range), dim=-1).view(-1, 2)
    
    # Discretize time
    t_start = st_x_cum[0, -1, -1].item()
    t_start = math.floor(t_start) + 1
    t_stop = math.ceil(total_time)
    n_steps = (t_stop - t_start) * num_frames_per_time_step + 1
    t_range = torch.linspace(t_start, t_stop, n_steps)

    # Calculate intensity
    background = model.background.unsqueeze(0).cpu().detach()

    # Sample model parameters
    _, w_i, b_i, inv_var = model(st_x.to(device))
    w_i = w_i.cpu().detach()
    b_i = b_i.cpu().detach()
    inv_var = inv_var.cpu().detach()
    
    # Convert to history
    his_st = torch.vstack((st_x[0], st_y.squeeze())).numpy()
    his_st_cum = torch.vstack((st_x_cum[0], st_y_cum.squeeze())).numpy()

    for t in tqdm(t_range, leave=verbose):
        i = sum(st_x_cum[:, -1, -1] <= t) - 1  # index of corresponding history events

        st_x_ = st_x[i:i + 1]
        w_i_ = w_i[i:i + 1]
        b_i_ = b_i[i:i + 1]
        inv_var_ = inv_var[i:i + 1]

        t_ = t - st_x_cum[i:i + 1, -1, -1]  # time since lastest event
        t_ = (t_ - biases[-1]) / scales[-1]

        # Calculate temporal intensity
        t_cum = torch.cumsum(st_x_[..., -1], -1)
        tn_ti = t_cum[..., -1:] - t_cum  # t_n - t_i
        tn_ti = torch.cat((tn_ti, torch.zeros(1, config.num_points)), -1)
        t_ti = tn_ti + t_

        lamb_t = t_intensity(w_i_, b_i_, t_ti) / np.prod(scales)

        # Calculate spatial intensity
        N = len(s_grids)  # number of grid points

        s_x_ = torch.cat((st_x_[..., :-1], background), 1).repeat(N, 1, 1)
        s_diff = s_grids.unsqueeze(1) - s_x_
        lamb_s = s_intensity(w_i_.repeat(N, 1), b_i_.repeat(N, 1), t_ti.repeat(N, 1),
                             s_diff, inv_var_.repeat(N, 1, 1))

        lamb = (lamb_s * lamb_t).view(x_nstep, y_nstep)
        lambs.append(lamb.numpy())
    lambs = np.array(lambs)

    x_range = x_range.numpy() * scales[0] + biases[0]
    y_range = y_range.numpy() * scales[1] + biases[1]
    t_range = t_range.numpy()

    return lambs, x_range, y_range, t_range, his_st_cum[:, :2], his_st_cum[:, 2]


"""==============================================================================
The code below is added by Tian You
=============================================================================="""

def get_model_output(model, st_x, device):
    background = model.background.cpu().detach()
    _, w_i, b_i, inv_var = model(st_x.reshape([1,st_x.shape[0],-1]).to(device))
    w_i  = w_i.cpu().detach()
    b_i  = b_i.cpu().detach()
    inv_var = inv_var.cpu().detach()
    params = (background, w_i, b_i, inv_var)
    return params

def calc_s_kernel(s_diff, inv_var):
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5*g2)/(2*np.pi)
    return g2

def s_kernel_mesh(params, seq, mesh_centre, scales=np.ones(2), biases=np.zeros(2)):

    # get the parameters    
    background, _, _, inv_var = params #get_model_params(model, st_x.reshape([1,st_x.shape[0],-1]), device)

    # get the input data and scale it
    st_x, _, _, _, _ = seq
    bias_xy = torch.tensor(biases[0:2])
    scales_xy = torch.tensor(scales[0:2])
    s_grids = (mesh_centre - bias_xy) / scales_xy
    
    N = len(s_grids) # number of grid points
    st_x = torch.cat((st_x[..., :-1], background), 0).unsqueeze(0).repeat(N, 1, 1)
    s_diff = s_grids.unsqueeze(1) - st_x
    s_kernel = calc_s_kernel(s_diff, inv_var.repeat(N, 1, 1))    

    #print(s_kernel[0])

    return s_kernel

def t_kernel_mesh(params, seq, time_ahead, scales=np.ones(1), biases=np.zeros(1)):
    
    background, _, b_i, _ = params
    num_points = background.shape[0]

    _, _, st_x_cum, _, _ = seq

    tn_ti = torch.cat((st_x_cum[-1,-1]-st_x_cum[:,2], torch.zeros(num_points)), 0)
    t_ti_scaled = torch.sub(torch.add(tn_ti, time_ahead), biases) /scales
    t_kernel = torch.exp(-b_i*t_ti_scaled)

    #print(-b_i*t_ti_scaled)

    return t_kernel

def t_integral_mesh(params, seq, time_period, scales=np.ones(1), biases=np.zeros(1)):
    background, _, b_i, _ = params
    num_points = background.shape[0]

    _, _, st_x_cum, _, _ = seq

    tp, tq = time_period

    tn_ti = torch.cat((st_x_cum[-1,-1]-st_x_cum[:,2], torch.zeros(num_points)), 0)
    print(tp)
    tp_ti_scaled = torch.sub(torch.add(tn_ti, tp), biases) /scales
    tq_ti_scaled = torch.sub(torch.add(tn_ti, tq), biases) /scales

    t_integral = 1/b_i * (torch.exp(-b_i*tp_ti_scaled) - torch.exp(-b_i*tq_ti_scaled))

    return t_integral


def calc_intensity_mesh(params, seq, mesh_centre, time_ahead, scales=np.ones(3), biases=np.zeros(3)):
    _, w_i, _, _ = params
    s_kernel = s_kernel_mesh(params, seq, mesh_centre, scales[:2], biases[:2])
    t_kernel = t_kernel_mesh(params, seq, time_ahead, scales[-1], biases[-1])
    lamb_st = torch.sum(s_kernel * t_kernel * w_i, -1).numpy()
    return lamb_st


def calc_expected_event_mesh(params, seq, mesh_centre, mesh_area, time_period, scales=np.ones(3), biases=np.zeros(3)):
    _, w_i, _, _ = params

    s_kernel = s_kernel_mesh(params, seq, mesh_centre, scales[:2], biases[:2])
    s_integral = s_kernel * mesh_area / np.product(scales[:2])
    print(s_kernel[0])
    t_integral = t_integral_mesh(params, seq, time_period, scales[-1], biases[-1])
    expected_event_count = torch.sum(s_integral * t_integral * w_i, -1).numpy()
    
    return expected_event_count



# The following is adapted from the Jupyter notebook '2. Deep-STPP Visualization'

# maybe the inference limit should be outside of (training) config


def pred_next_old(seq, device, config, scales=np.ones(3), biases=np.zeros(3), params=None):

    if params is None:
        background, w_i, b_i, _ = get_model_output(model, seq, device)
    else:
        background, w_i, b_i, _ = params

    num_points = background.shape[0]

    _, _, st_x_cum, _, _ = seq

    tp, tq = time_period

    tn_ti = torch.cat((st_x_cum[-1,-1]-st_x_cum[:,2], torch.zeros(num_points)), 0)
    tp_ti_scaled = torch.sub(torch.add(tn_ti, tp), biases) /scales
    tq_ti_scaled = torch.sub(torch.add(tn_ti, tq), biases) /scales

    t_integral = 1/b_i * (torch.exp(-b_i*tp_ti_scaled) - torch.exp(-b_i*tq_ti_scaled))
    
    t_intensity = 1


    return 1


def evaluate(model, scales, biases, testdata, device, config, params):
    model.eval()
    st_preds = []
    st_ys = []
    lookahead = config.lookahead

    print(lookahead)

    data = testdata
    #for index, data in enumerate(test_loader):
    #    if index == 0:
            #print(index, len(data))
    if len(data) == 5:
        st_x, _, st_x_cum, st_y, _ = data
    else:
        st_x, st_y = data

    batch_size = 1 #st_x.shape[0]
    st_pred = torch.zeros((batch_size, lookahead, 3), dtype=torch.float).to(device)
    #background = model.background.cpu().detach()
    
    for l in range(lookahead):

        background, w_i, b_i, _ = params

        #_, w_i, b_i, _ = model(st_x.reshape([1,st_x.shape[0],-1]).to(device))
        #w_i  = w_i.cpu().detach()
        #b_i  = b_i.cpu().detach()
        
        st_pred_1step = np.zeros((batch_size, 1, 3))
        
        t_cum = torch.cumsum(st_x[..., 2], -1)
        t_last = st_x_cum[-1,-1]
        tn_ti_2 = (t_last - st_x_cum[:,-1])/scales[-1]
        

        tn_ti = t_cum[..., -1:] - t_cum # t_n - t_i

        print(tn_ti, '\n', tn_ti_2)

        tn_ti = torch.cat((tn_ti, torch.zeros(config.num_points)), -1)
        
        # Time inference: integrate via linear interpolation
        limit = config.infer_limit * scales[-1]
        ts = torch.arange(0, limit, limit * 1.0 / config.infer_nstep)
        fts = [torch.exp(log_ft(t + tn_ti, tn_ti, w_i, b_i)) for t in ts]
        fts = torch.stack(fts) / sum(fts, 0) # normalize probability 
        predict_t = torch.sum(ts.unsqueeze(-1) * fts, 0) # expectation
        st_pred_1step[:, 0, 2] = predict_t.numpy()
        
        # Space Inference            
        v_i = w_i * torch.exp(-b_i * (tn_ti + predict_t.unsqueeze(-1)))
        v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1) # normalize
        v_i = v_i.unsqueeze(-1).numpy()
        
        space = torch.cat((st_x[..., :-1], background), 0)
        st_pred_1step[:, 0, :2] = np.sum(v_i * space.cpu().detach().numpy(), 1)
        
        st_pred_1step = torch.tensor(st_pred_1step, dtype=torch.float)
        
        st_pred[:, l, :] = st_pred_1step[:, 0, :]
        st_pred[:, l, -1] =  st_pred[:, l, -1]*scales + t_last
        #print(st_x[1:, :])
        #print(st_pred_1step[0])
        st_x = torch.cat((st_x[1:, :], st_pred_1step[0] ), 0)

        st_preds.append(st_pred)
        st_ys.append(st_y)

    '''
    st_preds = torch.cat(st_preds, dim=0).cpu().detach().numpy()
    st_y = torch.cat(st_ys, dim=0).cpu().detach().numpy()
    outputs = np.zeros(st_y.shape)
    targets = np.zeros(st_y.shape)

    
    for i in range(st_y.shape[0]):
        outputs[i] = st_scaler.inverse_transform(st_preds[i])
        targets[i] = st_scaler.inverse_transform(st_y[i])

    # Evaluate the performance using RMSE
    space_rmse = [np.mean([np.sqrt(MSE(outputs[:, i, :2], targets[:, i, :2])) for i in range(la)]) for la in
                  [1, ]] #[1, 5, 10, 20, 40]]
    time_rmse = [np.mean([np.sqrt(MSE(outputs[:, i, 2], targets[:, i, 2])) for i in range(la)]) for la in
                 [1, ]] #[1, 5, 10, 20, 40]]
    total_rmse = [np.mean([np.sqrt(MSE(outputs[:, i, :], targets[:, i, :])) for i in range(la)]) for la in
                  [1, ]] #[1, 5, 10, 20, 40]]

    logger.info(f"The RMSE for space is {space_rmse}")
    logger.info(f"The RMSE for time is {time_rmse}")
    logger.info(f"The 3-tuple RMSE  is {total_rmse}")
    '''

    return st_preds, st_ys #, [space_rmse, time_rmse, total_rmse]