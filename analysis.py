from cProfile import label
import os
import pandas as pd
from quant import *
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns


file_path = 'outputs/mobv2_grid_search'

BITs = [8]
Q_DIVs = [127.0, 127.1, 127.2, 127.3, 127.4, 127.5, 127.6, 127.7, 127.8, 127.9, 128.0, 128.1, 128.5, 200.0]
output_dir = f'outputs/vis/degradation'
os.makedirs(output_dir, exist_ok=True)
# QUANT_LAYER = 'dw1'
for BIT in BITs:
    for Q_DIV in Q_DIVs:
        print(f'Processing BIT {BIT}, Q_DIV {Q_DIV}')
        dfs = []
        for k, quant_layer in mbv2_map.items():
            filename = f'{quant_layer}_{BIT}_{Q_DIV}.csv'
            df = pd.read_csv(os.path.join(file_path, filename))
            dfs.append(df)
        sns.set(rc = {'figure.figsize':(15,8)})
        dfs = pd.concat(dfs, ignore_index=True)
        plt.xticks(rotation=45)
        plt.grid(True, which='both', ls='dashed')
        ax = sns.lineplot(x='Quant Layer', y="Top1 Diff", data=dfs, marker='o', label=f'{BIT}bit_{Q_DIV}')
        # plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{BIT}_{Q_DIV}.png'), dpi=300)
