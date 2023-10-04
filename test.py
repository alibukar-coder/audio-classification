import os
import gradio as gr
import numpy as np

TITLE = "Voice-based Age Verification"
DESP = ("Press the record button and say - I am happy or I am hungry")
port = int(os.environ.get('PORT', 8000))

def reverse_audio(audio):
    sr, data = audio
    return (sr, np.flipud(data))

def trial(input):
    # sr, data = audio
    return input


demo = gr.Interface(fn=reverse_audio, 
                    inputs="microphone",#"microphone", 
                    outputs="audio",#"audio", 
                    title=TITLE, allow_flagging="never",
                    description=DESP,
                    cache_examples=True)

if __name__ == "__main__":
    # demo.launch(debug=True)
    # demo.launch(share=True)
    
    demo.launch(server_name="0.0.0.0", server_port=port, debug=True)
