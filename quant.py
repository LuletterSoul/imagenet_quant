import torch 
from torch import nn

mbv2_map = {
    # Input Conv 1
    'features.0.0': 'conv1',

    # Bottleneck 1
    'features.1.conv.0': 'dw1',
    'features.1.conv.1': 'pw1_1',

    # Bottleneck 2
    'features.2.conv.0': 'pw2_1',
    'features.2.conv.1': 'dw2',
    'features.2.conv.2': 'pw2_2',

    # Bottleneck 3
    'features.3.conv.0': 'pw3_1',
    'features.3.conv.1': 'dw3',
    'features.3.conv.2': 'pw3_2',

    # Bottleneck 4
    'features.4.conv.0': 'pw4_1',
    'features.4.conv.1': 'dw4',
    'features.4.conv.2': 'pw4_2',

    # Bottleneck 5
    'features.5.conv.0': 'pw5_1',
    'features.5.conv.1': 'dw5',
    'features.5.conv.2': 'pw5_2',

    # Bottleneck 6
    'features.6.conv.0': 'pw6_1',
    'features.6.conv.1': 'dw6',
    'features.6.conv.2': 'pw6_2',


    # Bottleneck 7
    'features.7.conv.0': 'pw7_1',
    'features.7.conv.1': 'dw7',
    'features.7.conv.2': 'pw7_2',


    # Bottleneck 8
    'features.8.conv.0': 'pw8_1',
    'features.8.conv.1': 'dw8',
    'features.8.conv.2': 'pw8_2',

    # Bottleneck 9
    'features.9.conv.0': 'pw9_1',
    'features.9.conv.1': 'dw9',
    'features.9.conv.2': 'pw9_2',


    # Bottleneck 10
    'features.10.conv.0': 'pw10_1',
    'features.10.conv.1': 'dw10',
    'features.10.conv.2': 'pw10_2',

    # Bottleneck 11
    'features.11.conv.0': 'pw11_1',
    'features.11.conv.1': 'dw11',
    'features.11.conv.2': 'pw11_2',

    # Bottleneck 12
    'features.12.conv.0': 'pw12_1',
    'features.12.conv.1': 'dw12',
    'features.12.conv.2': 'pw12_2',


    # Bottleneck 13
    'features.13.conv.0': 'pw13_1',
    'features.13.conv.1': 'dw13',
    'features.13.conv.2': 'pw13_2',

    # Bottleneck 14
    'features.14.conv.0': 'pw14_1',
    'features.14.conv.1': 'dw14',
    'features.14.conv.2': 'pw14_2',

    # Bottleneck 15
    'features.15.conv.0': 'pw15_1',
    'features.15.conv.1': 'dw15',
    'features.15.conv.2': 'pw15_2',


    # Bottleneck 16
    'features.16.conv.0': 'pw16_1',
    'features.16.conv.1': 'dw16',
    'features.16.conv.2': 'pw16_2',


    # Bottleneck 17
    'features.17.conv.0': 'pw17_1',
    'features.17.conv.1': 'dw17',
    'features.17.conv.2': 'pw17_2',


    # Output Conv 2
    'features.18.0': 'conv2',

    # Classifier
    'classifier.1': 'fc1',

}

def quant_8_bit(model : nn.Module, args):
    layer = args.quant_layer
    BIT = args.BIT
    Q_DIV = args.Q_DIV
    Q_MAX = 2 ** (BIT - 1) - 1
    Q_MIN = - 2 ** (BIT - 1)
    print(f'Quant {BIT} bit, Q_MAX {Q_MAX}, Q_MIN {Q_MIN}')
    for name, p in model.named_parameters():
        if layer in name:
            print(f'Quant: {args.quant_rename_layer}')
            w = p.data 
            if w.dim() == 4:
                max_val = torch.amax(w.abs(), dim=(1, 2, 3), keepdim=True)
                min_val = torch.amin(w, dim=(1,2,3))
            else:
                max_val,_ = torch.max(w.abs(), dim=-1, keepdim=True)
                min_val,_ = torch.min(w.abs(), dim=-1, keepdim=True)
            # scale = 2 * max_val / (Q_MAX - Q_MIN)
            scale = max_val / Q_DIV
            p.data.div_(scale).round_().clamp_(Q_MIN, Q_MAX).mul_(scale)
    return model
