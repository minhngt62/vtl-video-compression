import torch
from torch import nn, Tensor
from dahuffman import HuffmanCodec

import numpy as np

def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
           
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    new_t = t_min + scale * quant_t
    return quant_t, new_t

def quantize_weights(model: nn.Module, quant_bit=8, quant_axis=0):
    # model quantization
    cur_ckt = model.state_dict()
    quant_weight_lst = []
    for k,v in cur_ckt.items():
        large_tf = (v.dim() in {2,4} and 'bias' not in k)
        quant_v, new_v = quantize_per_tensor(v, quant_bit, quant_axis if large_tf else -1)
        valid_quant_v = quant_v[v!=0] # only include non-zero weights
        quant_weight_lst.append(valid_quant_v.flatten())
        cur_ckt[k] = new_v
    return cur_ckt, quant_weight_lst


def entropy_encoding(model: nn.Module, quant_bit=8, quant_axis=0):
    _, quant_weight_lst = quantize_weights(model, quant_bit, quant_axis)

    # build symbols
    cat_param = torch.cat(quant_weight_lst)
    input_code_list = cat_param.tolist()
    
    # generate HuffmanCoding table
    codec = HuffmanCodec.from_data(input_code_list)
    return codec # can use pickle to save