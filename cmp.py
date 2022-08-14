import torch
import pandas as pd

original = 'outputs/mobv2/preds.csv'
quant = 'outputs/mobv2_quant/preds.csv'

original : pd.DataFrame = pd.read_csv(original)
quant : pd.DataFrame = pd.read_csv(quant)

diff = quant[quant['pred']!=original['pred']]

output_dir = 'outputs/diff.csv'

diff.to_csv(output_dir, index=False)



