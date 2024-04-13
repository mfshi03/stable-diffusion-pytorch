import torch
from . import Tokenizer, CLIP, Encoder, Decoder, Diffusion, OriginalDiffusion
from . import util
import warnings


def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace('_proj_weight', '_proj.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace('_proj_bias', '_proj.bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(("Given checkpoint data were modified dynamically by make_compatible"
                       " function on model_loader.py. Maybe this happened because you're"
                       " running newer codes with older checkpoint files. This behavior"
                       " (modify old checkpoints and notify rather than throw an error)"
                       " will be removed soon, so please download latest checkpoints file."))

    return state_dict

def load_clip(device):
    state_dict = torch.load(util.get_file_path('ckpt/clip.pt'))
    state_dict = make_compatible(state_dict)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict)
    return clip

def load_encoder(device):
    state_dict = torch.load(util.get_file_path('ckpt/encoder.pt'))
    state_dict = make_compatible(state_dict)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict)
    return encoder

def load_decoder(device):
    state_dict = torch.load(util.get_file_path('ckpt/decoder.pt'))
    state_dict = make_compatible(state_dict)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict)
    return decoder


def load_diffusion_original(device):
    state_dict = torch.load(util.get_file_path('ckpt/diffusion.pt'))
    state_dict = make_compatible(state_dict)

    diffusion = OriginalDiffusion().to(device)
    diffusion.load_state_dict(state_dict)
    return diffusion

def load_diffusion_new(device):
    state_dict = torch.load(util.get_file_path('ckpt/diffusion.pt'))
    state_dict = make_compatible(state_dict)

    diffusion = Diffusion().to(device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if "conv" in key:
            print(key)
            new_state_dict[key.replace(".weight", ".conv.weight")] = value
            out_channels, in_channels, kernel_size, _ = value.shape
            new_state_dict[key.replace('.weight', '.depthwise.weight')] = torch.rand(in_channels, 1, kernel_size, kernel_size) 
            new_state_dict[key.replace('.weight', '.pointwise.weight')] = torch.zeros(out_channels, in_channels, 1, 1) 
        else:
            new_state_dict[key] = value

    diffusion.load_state_dict(new_state_dict)

    return diffusion

def preload_models(device, original=False):
    return {
        'clip': load_clip(device),
        'encoder': load_encoder(device),
        'decoder': load_decoder(device),
        'diffusion': load_diffusion_original(device) if original else load_diffusion_new(device),
    }