from torch import nn

def params_to_prune(model: nn.Module):
    param_list = []
    for k,v in model.named_parameters():
        if 'weight' in k:
            if 'stem' in k:
                stem_ind = int(k.split('.')[1])
                param_list.append(model.stem[stem_ind])
            elif 'layers' in k[:6] and 'conv' in k:
                layer_ind = int(k.split('.')[1])
                param_list.append(model.layers[layer_ind].conv.conv)
    param_to_prune = [(ele, 'weight') for ele in param_list]
    return param_to_prune