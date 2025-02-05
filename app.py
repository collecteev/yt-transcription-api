from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
from openai import AsyncOpenAI
import os
from auth import require_custom_authentication
from dotenv import load_dotenv
import logging
import asyncio
import tiktoken

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API setup
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_youtube_id(url):
    """Extracts the YouTube video ID from a URL."""
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def process_transcript(video_id):
    """Fetches transcript from YouTube."""
    proxy = os.getenv("PROXY")
    transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies={"http": proxy, "https": proxy})
    return ' '.join([entry['text'] for entry in transcript])

def chunk_text(text, max_tokens=16000):
    """Splits text into chunks that fit within GPT-4o-mini’s token limit."""
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    words = text.split()
    chunks, current_chunk = [], []
    current_token_count = 0

    for word in words:
        word_token_count = len(tokenizer.encode(word + " "))
        if current_token_count + word_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_token_count = [], 0
        current_chunk.append(word)
        current_token_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def process_chunk(chunk):
    """Formats transcript using OpenAI GPT-4o-mini."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Format and punctuate the transcript."},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {str(e)}"

async def improve_text_with_gpt4(text):
    """Improves transcript formatting using GPT-4o-mini."""
    chunks = chunk_text(text)
    tasks = [process_chunk(chunk) for chunk in chunks]
    improved_chunks = await asyncio.gather(*tasks)
    return ' '.join(improved_chunks)

@app.route('/transcribe', methods=['POST'])
@require_custom_authentication
def transcribe():
    youtube_url = request.json.get('url')
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    video_id = get_youtube_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Processing video: {video_id}")
        transcript_text = process_transcript(video_id)
        improved_text = asyncio.run(improve_text_with_gpt4(transcript_text))
        return jsonify({"result": improved_text})
    
    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
