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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path

# Constants
RESAMPLE_RATE = 8000
TITLE = "Voice-based Age Verification"
DESP = ("Press the record button and say 'I am happy or I am hungry'")
# port = int(os.environ.get('PORT', 8000))


def main(audio):
    rate, waveform = audio
    # try:
    #     rate, waveform = audio
    # except Exception as e:
    #     return f"Error processing audio: {e}" 

    # #load models
    # svm_classifier, rf_classifier, xgb_classifier = load_models()

    # #preprocess waveform
    # preprocessed = preprocess(waveform, rate)

    # #get features
    # features = get_features(preprocessed)

    # if features is None:
    #     return "Empty audio input"

    # #get prediction
    # prediction = make_prediction(features, (svm_classifier, rf_classifier, xgb_classifier))

    prediction = "Debug: Main"
    
    return prediction

demo = gr.Interface(
    fn=main,
    gr.Audio(source="microphone"),
    outputs="text",
    allow_flagging="never", 
    title=TITLE, 
    description=DESP,
)

if __name__ == "__main__":
    # demo.launch(debug=True)
    # demo.launch(share=True)
    port = int(os.environ.get('PORT', 8000))
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
