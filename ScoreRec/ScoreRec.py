import numpy as np
import pandas as pd
import math
import random
import argparse
import os
import logging
import time as Time
from utility import calculate_hit, extract_axis_1,gini_coefficient
from Modules_ori import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
from scipy import integrate

logging.getLogger().setLevel(logging.INFO)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="ScoreRec")
    parser.add_argument('--Model', type=str, default="ScoreRec",
                        help='model name ')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='zhihu',
                        help='yoochoose, zhihu, sports_and_outdoors ')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed ')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size ')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--device', type=int, default=0,
                        help='cuda device ')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate ')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='Noise intensity ')
    parser.add_argument("--z", default=3, type=int, help='Cutoff index of DFT')
    parser.add_argument('--eps_train', type=float, default=0.00001,
                        help='Time step for adding noise during training ')
    parser.add_argument('--eps_test', type=float, default=0.0001,
                        help='Time step for adding noise during sampling ')
    parser.add_argument('--rtol', type=float, default=0.0001,
                        help='Relative Tolerance ')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    parser.add_argument('--atol', type=float, default=0.00001,
                        help='Absolute Tolerance ')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser, mlp1, mlp2')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='')
    parser.add_argument('--num_head', type=int, default=1,
                        help='')
    parser.add_argument('--InfoNCE', type=str2bool, default=False,
                        help='contrast loss ') #no use
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='weight of loss ')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature of contrast  ')

    parser.add_argument('--l2_decay', type=float, default=1e-8,
                        help='l2 loss reg coef ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency ')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='type of optimizer.')

    parser.add_argument('--sampler', type=str, default='ode_sampler',
                        help='ode_sampler')

    return parser.parse_args()

args = parse_args()
print(args)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)


def marginal_prob_std(t, sigma=args.sigma):
    """ Calculate the standard deviation of the post-perturbation condition Gaussian distribution at any time t """
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma=args.sigma):
    """ Calculate the diffusion coefficient at any time t """
    return (sigma ** t).clone().detach()


class diffusion():
    def __init__(self, sigma=0.5, snr=0.16, score=None):
        self.sigma = sigma
        self.snr = snr
        self.score = score

    def my_score_p_losses(self, denoise_model, x_start, h, eps=args.eps_train):
        sigma_t = torch.rand(x_start.shape[0]) * (1. - eps) + eps
        sigma_t = sigma_t.to(device=x_start.device)
        z = torch.randn_like(x_start)
        std = marginal_prob_std(sigma_t).to(device=x_start.device)  # 128
        perturbed_x = x_start + z * std[:, None]
        score = denoise_model(perturbed_x, h, sigma_t)
        loss = self.get_obj_denoise(x_start, z, score)
        return loss.mean(), score

    def get_obj_denoise(self, x_start, x, score):
        target = (x_start - x) / self.sigma ** 2  #
        obj = (score - target) ** 2
        obj *= self.sigma ** 2  #
        return obj

    def score_eval_wrapper(self, denoise_model, sample, h, time_steps, shape, device):
        """A wrapper of the score-based model for us by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        score = denoise_model(sample, h, time_steps)
        return score.detach().cpu().numpy().reshape((-1,))

    def ode_func(self, t, x, shape, device, denoise_model, h):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t))
        score = self.score_eval_wrapper(denoise_model, x, h, time_steps, shape, device)
        self.score = score
        return -0.5 * (g ** 2) * score

    def sample_from_last(self, h, denoise_model, samper= "ode_sampler", eps=0.0001):

        batch_size = h.size(0)
        device = h.device
        t = torch.ones(batch_size).to(device=device)
        std = marginal_prob_std(t).to(device=device)
        if h.size(-1) > args.hidden_size:
            x = torch.randn(h.size(0), h.size(1) // 3, device=device)
        else:
            x = torch.randn_like(h)
        x = x * std[:, None]
        shape = x.shape
        if samper == "ode_sampler":
            # Black-box ode solver for the probability floe ode
            res = integrate.solve_ivp(
                self.ode_func,
                (1., eps),  # time span
                x.reshape(-1).cpu().numpy(),  # initial condition
                args=(shape, device, denoise_model, h),  # pass other arguments as a tuple
                rtol=args.rtol,
                atol=args.atol,
                method='RK45'  # Only pass `method`
            )
            x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
            # x = x + self.sigma ** 2 * torch.tensor(self.score, device=device).reshape(shape)
            return x.float()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TFSMSR(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, marginal_prob_std,
                 num_heads=1, z=3, num_layers=3):
        super(TFSMSR, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size * 3,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )

        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)

        self.ln_4 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ln_5 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.z = z // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.alpha = 0.9

        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        self.time_dim = hidden_size
        if args.InfoNCE:
            # InfoNCE head
            self.proj_history1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.proj_history2 = nn.Linear(self.hidden_size * 3, self.hidden_size)
            self.proj_target = nn.Linear(self.hidden_size, self.hidden_size)
        self.step_mlp = nn.Sequential(
            GaussianFourierProjection(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        self.h_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        self.position_embeddings = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size)
        )

        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size)
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
        self.marginal_prob_std = marginal_prob_std
        self.diff_net = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_size, self.hidden_size * 3),
        )
        self.cond_all = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.cond_short = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size)
            ]
        )
        self.cond_long = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size)
            ]
        )

        self.cond_att = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size)
            ]
        )
        self.cond_t = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size)
            ]
        )
        self.linear_s = nn.Sequential(
            nn.Linear(self.hidden_size, 3)
        )
        self.linear_l = nn.Sequential(
            nn.Linear(self.hidden_size, 3)
        )
        self.linear_a = nn.Sequential(
            nn.Linear(self.hidden_size, 3)
        )
        self.linears_x = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            ]
        )


    def forward(self, x, h, step):
        t = self.step_mlp(step)
        if self.diffuser_type == 'mlp1':
            if int(h.size(1)) > self.hidden_size:
                h = self.h_mlp(h)
            # print(x.dtype, h.dtype,t.dtype)  # Check the data types of x and h
            res = torch.cat((x, h, t), dim=1)
            # res += self.g(x)
            res = self.diffuser(res)
            hh = self.marginal_prob_std(step)[:, None]
            return res / hh  # 128 64

        elif self.diffuser_type == 'mlp2':
            if int(h.size(1)) > self.hidden_size:
                h = self.h_mlp(h)
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            hh = self.marginal_prob_std(step)[:, None]
            return res / hh  # 128 64

    def Emb_x(self, x):
        x = self.item_embeddings(x)
        return x

    def Frequency_Layer(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        low_pass = x[:]
        low_pass[:, self.z:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')  # BxSxH
        high_pass = input_tensor - low_pass

        long_term_h = self.dropout(low_pass)
        short_term_h = self.dropout(high_pass)

        long_term_h = self.ln_4(long_term_h + input_tensor)
        short_term_h = self.ln_5(short_term_h + input_tensor)

        return torch.cat((long_term_h, short_term_h), dim=-1)

    def MultiH(self, states, len_states, p):
        inputs_emb = self.item_embeddings(states)  # 128 10 64
        inputs_emb += self.position_embeddings(torch.flip(torch.arange(self.state_size), dims=[0]).to(self.device))
        seq = self.emb_dropout(inputs_emb)  # 128 10 64
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)  # 128 10 64
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)  # 128 10 64

        DFUI = self.Frequency_Layer(seq)
        DFUI *= mask

        mh_DFUI = torch.cat((ff_out, DFUI), dim=-1)  # Self_attention+LongInterest+ShortInterest
        state_hidden = extract_axis_1(mh_DFUI, len_states - 1)  # B 1 H
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1 - mask)  ##128 64
        return h

    def predict(self, states, len_states, Score_diff):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.position_embeddings(torch.flip(torch.arange(self.state_size), dims=[0]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        DFUI = self.Frequency_Layer(seq)
        DFUI *= mask
        mh_DFUI = torch.cat((ff_out, DFUI), dim=-1)

        state_hidden = extract_axis_1(mh_DFUI, len_states - 1)
        h = state_hidden.squeeze()
        # sample
        x = Score_diff.sample_from_last(h=h,
                                        denoise_model=self.forward,
                                        samper=args.sampler,
                                        eps=args.eps_test)

        test_item_emb = self.item_embeddings.weight
        prob = torch.matmul(x, test_item_emb.transpose(0, 1))
        return prob,test_item_emb


def evaluate(model, test_data, Score_diff, device, writer, VAL, itr, logdirtxt=None):

    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))
    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(
        eval_data['next'].values)
    num_total = len(seq)
    recommendations = {k: [] for k in topk}
    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = (seq[i * batch_size: (i + 1) * batch_size],
                                      len_seq[i * batch_size: (i + 1) * batch_size],
                                      target[i * batch_size: (i + 1) * batch_size])
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction,test_item_emb = model.predict(states, np.array(len_seq_b), Score_diff)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        sorted_list2 = sorted_list2

        for k in topk:
            batch_recommendations = topK[:, :k].tolist()
            recommendations[k].extend(batch_recommendations)
        
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase)
        total_purchase += batch_size

    hr_list = []
    ndcg_list = []

    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@' + str(topk[0]),
                                                                                   'NDCG@' + str(topk[0]),
                                                                                   'HR@' + str(topk[1]),
                                                                                   'NDCG@' + str(topk[1]),
                                                                                   'HR@' + str(topk[2]),
                                                                                   'NDCG@' + str(topk[2]),
                                                                                   'HR@' + str(topk[3]),
                                                                                   'NDCG@' + str(topk[3])))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0, 0])

        if i == 1:
            hr_20 = hr_purchase
    
    if VAL:
        writer.add_scalar(tag='Evaluation/hr_list5', scalar_value=hr_list[0], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list5', scalar_value=(ndcg_list[0]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list10', scalar_value=hr_list[1], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list10', scalar_value=(ndcg_list[1]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list20', scalar_value=hr_list[2], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list20', scalar_value=(ndcg_list[2]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list50', scalar_value=hr_list[3], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list50', scalar_value=(ndcg_list[3]), global_step=itr)
    else:
        metrics = {}
        raw_embeddings = test_item_emb.detach().cpu().numpy()
        # L2 norm
        norm = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10  
        item_embeddings = raw_embeddings / norm

        for k in topk:
            user_lists = recommendations[k]
            if not user_lists:
                continue
            gini = gini_coefficient(user_lists, item_embeddings.shape[0])

            metrics[k] = {
            'Gini': gini
            }

        print("Diversity Metrics:")
    
        for k in topk:
            print(f"Gini={metrics[k]['Gini']:.4f}")
            writer.add_scalar(tag='Test/gini_{}'.format(k), scalar_value=(metrics[k]['Gini']), global_step=itr)

        writer.add_scalar(tag='Test/hr_list5', scalar_value=hr_list[0], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list5', scalar_value=(ndcg_list[0]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list10', scalar_value=hr_list[1], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list10', scalar_value=(ndcg_list[1]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list20', scalar_value=hr_list[2], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list20', scalar_value=(ndcg_list[2]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list50', scalar_value=hr_list[3], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list50', scalar_value=(ndcg_list[3]), global_step=itr)

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0],
                                                                                                   (ndcg_list[0]),
                                                                                                   hr_list[1],
                                                                                                   (ndcg_list[1]),
                                                                                                   hr_list[2],
                                                                                                   (ndcg_list[2]),
                                                                                                   hr_list[3],
                                                                                                   (ndcg_list[3])))
    

    return hr_20

if __name__ == '__main__':

    logdir = "./logs/Model_{}_data_{}_batch_size_{}_lr_{}_sigma_{}_sampler_{}_rtol_{}_atol_{}_diffuser_type_{}_num_head_{}_InfoNCE_{}_alpha_{}_temperature_{}_c_{}".format(
        args.Model, args.data, args.batch_size, args.lr, args.sigma, args.sampler,
        args.rtol, args.atol, args.diffuser_type, args.num_head, args.InfoNCE, args.alpha, args.temperature, args.z)
    logdir_pth = "./pth/Model_{}_data_{}".format(args.Model, args.data)

    if not os.path.exists(logdir_pth):
        os.mkdir(logdir_pth)
    
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk = [5, 10, 20, 50]
    device = torch.device(args.device)

    model = TFSMSR(hidden_size=args.hidden_size,
                   item_num=item_num,
                   state_size=seq_size,
                   dropout=args.dropout_rate,
                   diffuser_type=args.diffuser_type,
                   device=device,
                   marginal_prob_std=marginal_prob_std,
                   num_heads=args.num_head,
                   z=args.z,
                   num_layers=args.num_layers)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    Score_diff = diffusion(sigma=args.sigma,
                           snr=0.16)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=20)
    model.to(device)
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    num_rows = train_data.shape[0]
    num_batches = int(num_rows / args.batch_size)
    for i in range(args.epoch):
        start_time = Time.time()
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())
            optimizer.zero_grad()
            seq = torch.tensor(seq).long()
            len_seq = torch.tensor(len_seq).long()
            target = torch.tensor(target).long()

            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            x_start = model.Emb_x(target)
            h = model.MultiH(seq, len_seq, args.p)
            SMSR_loss, _ = Score_diff.my_score_p_losses(model, x_start, h)
            if args.InfoNCE: # 实验中一直都是 False
                if int(h.size(0)) > args.hidden_size:
                    h_proj = F.normalize(model.proj_history2(h), dim=-1)  # [B, contrast_dim]
                else:
                    h_proj = F.normalize(model.proj_history1(h), dim=-1)  # [B, contrast_dim]
                t_proj = F.normalize(model.proj_target(x_start), dim=-1)  # [B, contrast_dim]
                sim_matrix = torch.mm(h_proj, t_proj.T)  # [B, B]
                labels = torch.arange(args.batch_size, device=device)
                # InfoNCE loss
                temperature = args.temperature
                contrast_loss = F.cross_entropy(
                    sim_matrix / temperature,
                    labels)
                # print(SMSR_loss,contrast_loss)
                total_loss = args.alpha * SMSR_loss + (1 - args.alpha) * contrast_loss
            else:
                total_loss = SMSR_loss
            total_loss.backward()
            optimizer.step()

        # # save model
        # if i % 10 == 0:
        #     model_save_path = os.path.join(logdir_pth, f"model_epoch_{i}.pth")
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"Model saved at epoch {i} to {model_save_path}")

        # scheduler.step()
        if args.report_epoch:
            model.eval()
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(
                    total_loss) + "Time cost: " + Time.strftime(
                    "%H: %M: %S", Time.gmtime(Time.time() - start_time)))
            if (i + 1) % 10 == 0:
                print('-------------------------- VAL PHRASE --------------------------')
                VAL = True
                _ = evaluate(model, 'val_data.df', Score_diff, device, writer, VAL, i)
                print('-------------------------- TEST PHRASE -------------------------')
                VAL = False
                eval_start = Time.time()
                _ = evaluate(model, 'test_data.df', Score_diff, device, writer, VAL, i)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))
                print('----------------------------------------------------------------')
