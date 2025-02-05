from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import re
import openai
import os
from auth import require_custom_authentication
from dotenv import load_dotenv
import logging
import tempfile

load_dotenv()

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Proxy setup
PROXY = os.getenv("PROXY")

# Whisper API client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_youtube_id(url):
    """Extracts video ID from YouTube URL."""
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def process_transcript(video_id):
    """Retrieves YouTube transcript if available."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies={"http": PROXY, "https": PROXY})
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        logger.warning(f"Transcript unavailable for {video_id}: {e}")
        return None  # If transcript fails, we use Whisper AI fallback

def download_audio(video_url):
    """Downloads audio using yt-dlp."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '-',
            'quiet': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        }
        
        if PROXY:
            ydl_opts['proxy'] = PROXY

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            ydl_opts['outtmpl'] = tmpfile.name
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            return tmpfile.name  # Return path to downloaded file

    except Exception as e:
        logger.error(f"Audio download failed: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper AI."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        logger.error(f"Whisper AI transcription failed: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
@require_custom_authentication
def transcribe():
    """Handles transcription requests."""
    youtube_url = request.json.get('url')
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    video_id = get_youtube_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    logger.info(f"Processing video: {video_id}")

    # Try to get transcript from YouTube first
    transcript_text = process_transcript(video_id)

    # If transcript is missing, fallback to Whisper AI
    if not transcript_text:
        logger.info(f"Using Whisper AI for {video_id}")
        audio_path = download_audio(youtube_url)

        if not audio_path:
            return jsonify({"error": "Could not retrieve audio for Whisper AI."}), 500

        transcript_text = transcribe_audio(audio_path)

    if not transcript_text:
        return jsonify({"error": "Transcription failed."}), 500

    return jsonify({"result": transcript_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
