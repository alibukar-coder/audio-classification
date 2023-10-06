# -*- coding: utf-8 -*-
"""
Voice age classifier using MFCC features and ensemble models.
Author: Ali Bukar
Date:   25th September 2023

"""

import os
import gradio as gr
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pickle
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path

# Constants
RESAMPLE_RATE = 8000
TITLE = "Voice-based Age Verification"
DESP = ("Press the record button and say: I am happy or I am hungry or any other sentence :-)")
port = int(os.environ.get('PORT', 8000))

# Define a function to load the classifiers
def load_models():
    """Load trained models"""

    with open('svm_classifier.pkl', 'rb') as svm_file:
        svm_model = pickle.load(svm_file)

    with open('rf_classifier.pkl', 'rb') as rf_file:
        rf_model = pickle.load(rf_file)

    with open('xg_classifier.pkl', 'rb') as xg_file:
        xg_model = pickle.load(xg_file)

    print('Debug Models Loaded!!!')

    return svm_model, rf_model, xg_model

def get_label(res_val):
    result = int(res_val[0])
    if result == 1:
        return 'Adult'
    else:
        return 'Child'

def preprocess(waveform, sample_rate):
    """Resample and extract relevant section of audio"""
    
    #convert and reshape waveform
    waveform = waveform.astype(np.float32)
    waveform = waveform.reshape(1, -1)

    # Convert the reshaped NumPy array to a PyTorch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Resample waveform
    resampled = T.Resample(sample_rate, RESAMPLE_RATE, dtype=waveform.dtype)(waveform)

    # Convert the start and end times to sample indices
    end_sample = int(0.9375*RESAMPLE_RATE)
    
    # Extract first second 
    extracted = resampled[:, 0:end_sample]
    
    return extracted

def get_features(audio):
    """Extract MFCC features"""
    
    mfcc_transform = T.MFCC(
        sample_rate=RESAMPLE_RATE,
        n_mfcc=256,
        melkwargs={
        'n_fft': 2048,
        'n_mels': 256,
        'hop_length': 512,
        'mel_scale': 'htk',
        }
    )
    features  = mfcc_transform(audio)
    features  = features.reshape(-1).numpy()
    features  = features.reshape(1, -1)

    #check to make sure that the waveform is not empty
    if len(features) == 0:
        return None

    return features

def make_prediction(features, models):
    """Make prediction using ensemble of models"""

    svm_pred = models[0].predict(features)
    rf_pred  = models[1].predict(features)
    xgb_pred = models[2].predict(features)

    # Collect all predictions
    predictions = [svm_pred[0], rf_pred[0], xgb_pred[0]]
    countt = Counter(predictions)
    voted_res = countt.most_common()

    if voted_res == 1:
        return 'Adult voice detected'
    else:
        return 'Child voice detected'

def main(audio):
    print('Debug: Main')
    try:
        rate, waveform = audio
    except Exception as e:
        return f"Error processing audio: {e}" 

    #load models
    svm_classifier, rf_classifier, xgb_classifier = load_models()

    #preprocess waveform
    preprocessed = preprocess(waveform, rate)

    #get features
    features = get_features(preprocessed)

    if features is None:
        return "Empty audio input"

    #get prediction
    prediction = make_prediction(features, (svm_classifier, rf_classifier, xgb_classifier))
    
    return prediction

demo = gr.Interface(
    main,
    gr.Audio(source="microphone"),
    outputs="text",
    allow_flagging="never", 
    title=TITLE, 
    description=DESP,
)

if __name__ == "__main__":
    # demo.launch(debug=True)
    # demo.launch(share=True)
    
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
