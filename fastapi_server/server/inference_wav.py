import os, argparse
import torch
import audiofile
from transformers import Wav2Vec2ForCTC
from .score_model import MLP
import numpy as np


def inference_wav(wav_filepath):
    lang = 'en'  # Static language parameter
    label_type1 = 'pron'  # Static label type parameter
    label_type2 = 'articulation'  # Static label type parameter
    dir_model = 'model_ckpt/'  # Static model directory parameter
    device = 'cpu'  # Static device parameter (use 'cuda' if GPU is available)
    audio_len_max = 200000  # Static audio length parameter

    # Determine the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, dir_model)
    dir_model = os.path.join(model_dir, f'lang_{lang}', f'{label_type1}_{label_type2}_checkpoint.pt')

    try:
        score_model = torch.load(dir_model, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}")
    score_model.eval()

    if lang == 'en':
        base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    print(f'{lang}, {label_type1}, {label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(device)
    x, sr = audiofile.read(wav_filepath)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], device=device).reshape(1, -1)
    feat_x = base_model(x, output_attentions=True, output_hidden_states=True, return_dict=True).hidden_states[-1]
    feat_x = torch.mean(feat_x, axis=1)

    pred_score = score_model(feat_x).cpu().detach().numpy()
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0][0]}')
    return pred_score[0][0]

def inference_wav2(wav_filepath):
    lang = 'en'  # Static language parameter
    label_type1 = 'pron'  # Static label type parameter
    label_type2 = 'prosody'  # Static label type parameter
    dir_model = 'model_ckpt/'  # Static model directory parameter
    device = 'cpu'  # Static device parameter (use 'cuda' if GPU is available)
    audio_len_max = 200000  # Static audio length parameter

    # Determine the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, dir_model)
    dir_model = os.path.join(model_dir, f'lang_{lang}', f'{label_type1}_{label_type2}_checkpoint.pt')

    try:
        score_model = torch.load(dir_model, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}")
    score_model.eval()

    if lang == 'en':
        base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    print(f'{lang}, {label_type1}, {label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(device)
    x, sr = audiofile.read(wav_filepath)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], device=device).reshape(1, -1)
    feat_x = base_model(x, output_attentions=True, output_hidden_states=True, return_dict=True).hidden_states[-1]
    feat_x = torch.mean(feat_x, axis=1)

    pred_score = score_model(feat_x).cpu().detach().numpy()
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0][0]}')
    return pred_score[0][0]