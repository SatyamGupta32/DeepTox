import os
import tempfile
import pickle
import gradio as gr
import speech_recognition as sr
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load your trained model and tokenizer
model = load_model("toxicity.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 100    # ensure this matches what the model was trained with
THRESHOLD = 0.2     # lower if model is conservative

def audio_to_text(audio_file):
    # audio_file is a filepath provided by Gradio; speech_recognition expects WAV/AIFF/FLAC
    # try to read directly; if format unsupported, return helpful error
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Audio not understood"
    except sr.RequestError:
        return "Speech service unavailable"
    except Exception as e:
        # Helpful fallback: tell user to upload WAV/AIFF/FLAC or convert
        return f"Error processing audio: {e}. Use WAV/AIFF/FLAC."

def score_comment_or_audio(comment, audio_file):
    transcription = ""
    if audio_file:
        transcription = audio_to_text(audio_file)
        # if transcription is an error message, return it
        if transcription.startswith("Error") or transcription in ("Audio not understood", "Speech service unavailable"):
            return transcription, ""
        comment = transcription

    if not comment or comment.strip() == "":
        return "No input provided", transcription

    # Tokenize & pad
    sequences = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH)

    # Get raw model outputs
    probs = model.predict(padded)
    # If single-output model, make it 2D for uniform handling
    probs = np.atleast_2d(probs)

    # DEBUG: show raw probabilities in output to inspect model behaviour
    probs_list = probs[0].tolist()
    debug_probs = ", ".join([f"{p:.4f}" for p in probs_list])

    # Build human readable result using threshold
    result_lines = []
    for i, p in enumerate(probs_list):
        label = "Toxic" if p > THRESHOLD else "Not Toxic"
        result_lines.append(f"Class {i+1}: {label} ({p:.4f})")
    result_text = "\n".join(result_lines)
    result_text += f"\n\nRaw probs: [{debug_probs}]"

    return result_text, transcription

interface = gr.Interface(
    fn=score_comment_or_audio,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type a comment here", label="Comment"),
        gr.Audio(type="filepath", label="Audio")
    ],
    outputs=[
        gr.Textbox(lines=8, label="Toxicity Prediction", interactive=False),
        gr.Textbox(lines=6, label="Transcription", interactive=False)
    ],
    title="Toxicity Classifier (debug)"
)

if __name__ == "__main__":
    interface.launch(share=True)





# import gradio as gr
# import speech_recognition as sr

# def audio_to_text(audio_file):
#     r = sr.Recognizer()
#     try:
#         with sr.AudioFile(audio_file) as source:
#             audio = r.record(source)
#         return r.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "Audio not understood"
#     except sr.RequestError:
#         return "Speech service unavailable"
#     except Exception as e:
#         return f"Error processing audio: {e}. Use WAV/AIFF/FLAC."

# # Gradio UI — only transcription display
# interface = gr.Interface(
#     fn=audio_to_text,
#     inputs=gr.Audio(type="filepath", label="Upload Audio"),
#     outputs=gr.Textbox(lines=6, label="Transcription", interactive=False),
#     title="Audio Transcription"
# )

# if __name__ == "__main__":
#     interface.launch(share=True)
