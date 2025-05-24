# 🎙️ Accent Detection from Video

This Streamlit app detects the speaker's accent from an uploaded video file using a fine-tuned Hugging Face model.

## 🚀 Features

* Upload video files (`.mp4`, `.mov`, `.avi`, `.mkv`)
* Extracts audio using `ffmpeg`
* Uses `HamzaSidhu786/speech-accent-detection` for accent classification
* Deployable via Streamlit

## 🧠 Model

Model used: [HamzaSidhu786/speech-accent-detection](https://huggingface.co/HamzaSidhu786/speech-accent-detection)
Framework: Hugging Face Transformers + PyTorch

## 🛠️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/accent-detection.git
   cd accent-detection
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

> ⚠️ Requires `ffmpeg` to be installed and available in your system PATH.

## 🧾 Example

1. Upload a video file.
2. Wait for audio extraction and model prediction.
3. View the detected accent.

## 📂 File Structure

```
├── app.py
├── README.md
├── requirements.txt
```
