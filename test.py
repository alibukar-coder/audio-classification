import os
import gradio as gr
import numpy as np
import pickle
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path

TITLE = "Voice-based Age Verification"
DESP = ("Press the record button and say - I am happy or I am hungry")
port = int(os.environ.get('PORT', 8000))

def load_models():
    """Load trained models"""

    with open('svm_classifier.pkl', 'rb') as svm_file:
        svm_model = pickle.load(svm_file)

    with open('rf_classifier.pkl', 'rb') as rf_file:
        rf_model = pickle.load(rf_file)

    with open('xgb_classifier.pkl', 'rb') as xgb_file:
        xgb_model = pickle.load(xgb_file)

    print('Debug Models Loaded!!!')

    return svm_model, rf_model, xgb_model
    
def reverse_audio(audio):
    print('Load Models!')
    all_models = load_models()
    sr, data = audio
    return (sr, np.flipud(data))

def trial(input):
    # sr, data = audio
    return input


demo = gr.Interface(reverse_audio, 
                    gr.Audio(source="microphone", type="numpy", label="Speak here..."),
                    outputs="audio", 
                    title=TITLE, allow_flagging="never",
                    description=DESP,
                    cache_examples=True)

if __name__ == "__main__":
    # demo.launch(debug=True)
    # demo.launch(share=True)
    
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
