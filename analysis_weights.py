from cProfile import label
import os
import pandas as pd
from quant import *
from pathlib import Path
import torchvision.models as models
from bn_fold import fuse_bn_recursively

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns

model : models.MobileNetV2 = models.MobileNetV2()
model.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'))
model = fuse_bn_recursively(model)
output_dir = 'outputs/vis/w&b'
os.makedirs(output_dir, exist_ok=True)

# layer = 'features.1.conv.0'
# rename = 'dw1'
# layer = 'features.2.conv.1'
# rename = 'dw2'
layer = 'features.2.conv.2'
rename = 'pw2_2'

def plot_hist(w, label, fig_name, color='r'):
    # axis = sns.histplot(data=w, label=fig_name, color=color, multiple='stack')
    axis = sns.histplot(data=w)
    axis.set_ylabel(label)
    # plt.clf()
    # axis.clear()

def plot_disthist(w, path, label):
    import matplotlib.pyplot as plt
    axis = plt.hist(w)
    # axis.set_ylabel(label)
    plt.savefig(path, dpi=300)
    plt.close()

for name, p in model.named_parameters():
        Q_DIV = 127.5
        # print(name)
        if layer in name:
            w = p.data.reshape(-1).detach().cpu().numpy()
            if 'weight' in name:
                # axis = sns.histplot(data=w)
                # axis.set_ylabel("Weights")
                # plt.savefig(os.path.join(output_dir, f'{rename}_weight.png'), dpi=300)
                # plt.close()
                max_val = torch.amax(p.data.abs(), dim=(1, 2, 3), keepdim=True)
                scale = max_val / Q_DIV
                p_ = p.clone()
                p_.data.div_(scale).round_().clamp_(-128, 127).mul_(scale)
                w_ = p_.data
                print(f'MSE Quant Loss: {F.mse_loss(p.data, w_) : .6f}')
                w_ = w_.reshape(-1).detach().cpu().numpy()
                df = pd.DataFrame({'Weight': w, 'Weight DeQuant': w_})
                plot_hist(df, 'Weights', 'Weights', color='r')
                plot_hist(df, 'Weights', 'Weights', color='b')
                plt.savefig(os.path.join(output_dir, f'{rename}_weight_hist.png'), dpi=300)
                plt.clf()
                sns.lineplot(data=w, label='Weight', marker='o')
                sns.lineplot(data=w_, label='Weight DeQuant', marker='^')
                plt.savefig(os.path.join(output_dir, f'{rename}_weight_line.png'), dpi=300)
                plt.clf()
                # plot_hist(w_, os.path.join(output_dir, f'{rename}_weight^.png'), 'Weights', 'Weights^', color='b')
            if 'bias' in name:
                max_val = torch.amax(p.data.abs(), dim=-1, keepdim=True)
                scale = max_val / Q_DIV
                w_ = p.clone().data.div_(scale).round_().clamp_(-128, 127).mul_(scale).reshape(-1).detach().cpu().numpy()
                df = pd.DataFrame({'Bias': w, 'Bias DeQuant': w_})
                plot_hist(df, label='Bias', fig_name='Bias', color='r')
                plt.savefig(os.path.join(output_dir, f'{rename}_bias_hist.png'), dpi=300)
                plt.clf()
                sns.lineplot(data=w, label='Bias', marker='o')
                sns.lineplot(data=w_, label='Bias DeQuant', marker='^')
                plt.savefig(os.path.join(output_dir, f'{rename}_bias_line.png'), dpi=300)
                # plot_hist(w_, os.path.join(output_dir, f'{rename}_bias^.png'), label='Bias', fig_name='Bias^', color='b')
                plt.clf()
