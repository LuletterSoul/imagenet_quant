import os

from quant import *
import subprocess
from tqdm import tqdm

BITs = [8]
Q_DIVs = [127, 127.1, 127.2, 127.3, 127.4, 127.5, 127.6, 127.7, 127.8, 127.9, 128, 128.1, 128.5, 200]
# Q_DIVs = [127]
with tqdm(total=len(BITs) * len(Q_DIVs) * len(mbv2_map)) as pbar:
    for BIT in BITs:
        for Q_DIV in Q_DIVs:
            i = 0
            for k, v in mbv2_map.items():
                try:
                    print(f'Processing BIT {BIT}, Q_DIV {Q_DIV}, Quant Layer {v}')
                    p = subprocess.Popen(['python', 
                    'main_mobv2.py', '--data', 'datasets/imagenet', 
                    '--pretrained', '-e', '-b', '1024', '-j' ,'64', '--output_dir', 'outputs/mobv2_grid_search', '--dist-backend', 'nccl', '--dist-url', 'tcp://127.0.0.1:6006'
                    , '--multiprocessing-distributed', '--world-size' , '1', '--rank', '0', '-q', '--BIT', str(BIT), '--Q_DIV', str(Q_DIV), '--quant_layer', str(k), '--quant_rename_layer', str(v)])
                    p = p.wait()
                except Exception as e:
                    pass
                finally:
                    pbar.update(1)