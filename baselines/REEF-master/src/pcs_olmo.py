import gc
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_model_save_path
import os

def release():
    torch.cuda.empty_cache()
    gc.collect()
    
def load_params(model_tag):
    model_name = "allenai/OLMo-2-1124-7B"


    model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_tag, trust_remote_code=True).eval()
    model = model.eval()
    params = torch.cat([model.state_dict()[key].view(-1) for key in model.state_dict().keys() if 'lm_head' not in key and 'embed' not in key]) 

    return params

def cal_cos(source_params, target_params, align_strategy='truncation'):
    len1, len2 = source_params.size()[0], target_params.size()[0]
    if len1 != len2:
        print(align_strategy)
        if align_strategy == 'random_sampling':
            if len1 > len2:
                indices = torch.randperm(len1)[:len2]
                source_params = source_params[indices]
            elif len2 > len1:
                indices = torch.randperm(len2)[:len1]
                target_params = target_params[indices]
        elif align_strategy == 'truncation':
            if len1 > len2:
                source_params = source_params[:len2]
            elif len2 > len1:
                target_params = target_params[:len1]
        elif align_strategy == 'padding':
            if len1 > len2:
                padding = torch.zeros(len1 - len2)
                target_params = torch.cat([target_params, padding], dim=0)
            elif len2 > len1:
                padding = torch.zeros(len2 - len1)
                source_params = torch.cat([source_params, padding], dim=0)

    
    source_params = source_params.detach().numpy()
    target_params = target_params.detach().numpy()
    cosine_similarity = 1 - cosine(source_params, target_params)

    return cosine_similarity

    
base_model = "stage1-step928000-tokens3893B"
source_params = load_params(base_model)
tmodel_tags = [
    "stage1-step1000-tokens5B",
    "stage1-step207000-tokens869B",
    "stage1-step310000-tokens1301B",
    "stage1-step413000-tokens1733B",
    "stage1-step516000-tokens2165B",
    "stage1-step619000-tokens2597B",
    "stage1-step722000-tokens3029B",
    "stage1-step825000-tokens3461B",
    "stage1-step928000-tokens3893B"
]
align_strategys = ['truncation']
for align_strategy in align_strategys:
    for tmodel_tag in tmodel_tags:
        target_params = load_params(tmodel_tag)
        cos_sim = cal_cos(source_params, target_params, align_strategy)
        
        print(f'PCS of {base_model} and {tmodel_tag} is: {cos_sim}')
        release()