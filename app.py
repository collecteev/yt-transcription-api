from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import re
import yt_dlp
import os
import logging
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai import OpenAIError
from auth import require_custom_authentication
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Proxy setup for YouTube transcript API
proxy_address = os.environ.get("PROXY")

# Path to cookies file
COOKIES_FILE = "cookies.txt"


def get_youtube_id(url):
    """Extracts video ID from YouTube URL."""
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None


def process_transcript(video_id):
    """Retrieves transcript using YouTubeTranscriptAPI."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies={"http": proxy_address, "https": proxy_address})
        full_text = ' '.join([entry['text'] for entry in transcript])
        return full_text
    except TranscriptsDisabled:
        logger.warning(f"Transcript unavailable for {video_id}, fallback to Whisper.")
        return None


def download_audio(video_url):
    """Downloads the audio using yt-dlp with cookies."""
    output_path = "audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookiefile': COOKIES_FILE,  # Use the cookies.txt file for authentication
        'quiet': False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except Exception as e:
        logger.error(f"Audio download failed: {e}")
        return None


async def process_chunk(chunk):
    """Processes a chunk of transcript text with OpenAI."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves text formatting and adds punctuation."},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"OpenAI API error: {str(e)}"


async def improve_text_with_gpt4(text):
    """Enhances the transcript using OpenAI GPT-4o-mini."""
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]  # Chunk to avoid token limit
    tasks = [process_chunk(chunk) for chunk in chunks]
    improved_chunks = await asyncio.gather(*tasks)
    return ' '.join(improved_chunks)


@app.route('/transcribe', methods=['POST'])
@require_custom_authentication
def transcribe():
    """API endpoint for transcribing YouTube videos."""
    youtube_url = request.json.get('url')
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    video_id = get_youtube_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Processing video: {video_id}")
        
        # Try to get transcript
        transcript_text = process_transcript(video_id)

        # If transcript unavailable, use Whisper AI
        if not transcript_text:
            logger.info(f"Using Whisper AI for {video_id}")
            audio_file = download_audio(youtube_url)

            if not audio_file:
                return jsonify({"error": "Could not retrieve audio for Whisper AI."}), 500
            
            with open(audio_file, "rb") as file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file
                )
            transcript_text = response.text

        # Improve transcript formatting
        improved_text = asyncio.run(improve_text_with_gpt4(transcript_text))

        return jsonify({"result": improved_text})

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
