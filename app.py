
import streamlit as st
import ffmpeg
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import tempfile
import os

# Function to extract audio
def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        st.error("Audio extraction failed.")
        st.stop()
    return audio_path

# Load model and feature extractor
@st.cache_resource
def load_model():
    model_name = "HamzaSidhu786/speech-accent-detection"
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, extractor

# Predict accent
def predict_accent(audio_path, model, extractor):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred_id]

# Streamlit UI
st.title("Accent Detection from Video")
st.write("Upload a video to detect the speaker's accent.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded.read())
        video_path = tmp_vid.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_aud:
        audio_path = tmp_aud.name

    st.info("Extracting audio...")
    extract_audio(video_path, audio_path)

    st.info("Loading model...")
    model, extractor = load_model()

    st.info("Detecting accent...")
    accent = predict_accent(audio_path, model, extractor)
    st.success(f"Detected Accent: **{accent}**")

    os.remove(video_path)
    os.remove(audio_path)
