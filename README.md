# 🎙️ Auralis
Auralis is an AI-powered transcription tool that converts audio and video into accurate text. Built on OpenAI Whisper, it supports multiple languages and works seamlessly for meetings, lectures, interviews, podcasts, and recordings with mixed-language conversations.

## ✨ Features
- 🎧 Transcribe **audio & video** files
- 🌍 Supports **multiple languages**
- ⚡ Fast & accurate transcription with Whisper
- 📝 Generates clean, readable transcripts
- 🖥️ Simple **command-line interface**

## Dependencies
pip install openai-whisper yt-dlp ffmpeg-python tqdm
Note: ffmpeg must be installed on your system and on PATH
* Windows (scoop): scoop install ffmpeg
* macOS (brew): brew install ffmpeg
* Linux (apt): sudo apt-get install ffmpeg

## Supported Models

1. tiny – Fastest, less accurate
2. base – Balanced speed/accuracy
3. small – Good accuracy, moderate speed
4. medium – High accuracy, slower
5. large – Best accuracy, resource-heavy

Notes
- The first run for a given model downloads weights (~30MB–3GB). Use a smaller model if you want speed.
- For best accuracy across languages, use --model large-v3 (slow, big), otherwise small/medium are good trade‑offs.

## 🎯 Usage
Transcribe a local video: 
python auralis.py --input path/to/video.mp4 --model small

Transcribe from a YouTube URL: 
python auralis.py --url https://www.youtube.com/watch?v=<id> --model base

<img width="1660" height="472" alt="image" src="https://github.com/user-attachments/assets/46fa6be4-f28c-44de-9510-06e54f2a424f" />

<img width="1738" height="821" alt="image" src="https://github.com/user-attachments/assets/e75d0bc0-1b43-4607-92d3-46a35e5c72ea" />

