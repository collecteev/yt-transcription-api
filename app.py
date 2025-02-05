from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
import yt_dlp
import openai
import os
from auth import require_custom_authentication
from dotenv import load_dotenv
import logging
import asyncio
import whisper
import ffmpeg

load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Proxy settings
PROXY = os.getenv("PROXY")

def get_youtube_id(url):
    """Extract video ID from YouTube URL."""
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def process_transcript(video_id):
    """Attempt to fetch subtitles from YouTube. Fallback to Whisper AI if unavailable."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies={"http": PROXY, "https": PROXY})
        full_text = ' '.join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        logger.warning(f"⚠️ No subtitles found. Falling back to Whisper AI... {e}")
        return process_whisper_transcription(video_id)

def download_audio(video_id):
    """Download YouTube audio without storing it on disk."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': '-',  # Direct output to stdout (memory)
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return result['url']
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def process_whisper_transcription(video_id):
    """Download & transcribe audio using Whisper AI."""
    audio_url = download_audio(video_id)
    
    if not audio_url:
        return "Error: Could not retrieve audio for Whisper AI."

    try:
        logger.info("🔹 Downloading and processing audio via Whisper AI...")
        audio = whisper.load_audio(audio_url)
        model = whisper.load_model("base")
        result = model.transcribe(audio)
        return result["text"]
    except Exception as e:
        logger.error(f"❌ Whisper AI transcription failed: {e}")
        return "Error: Whisper AI failed to transcribe the audio."

@app.route('/transcribe', methods=['POST'])
@require_custom_authentication
def transcribe():
    """Main API route for transcription."""
    youtube_url = request.json.get('url')
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    video_id = get_youtube_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Processing video: {video_id}")
        transcript_text = process_transcript(video_id)
        return jsonify({"result": transcript_text})
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
