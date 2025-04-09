# Index-TTS

Zero-shot text-to-speech system for voice cloning with just a short audio sample. Supports English and Chinese.

## Setup

```bash
# Environment
conda create -n index-tts python=3.10
conda activate index-tts
pip install -r requirements.txt
apt-get install ffmpeg

# Download models
bash ./download_models.sh
```

## Usage

### Web Interface
```bash
python webui.py
```

### Command Line
```bash
# Place reference audio in test_data/input.wav
python execute.py
```
