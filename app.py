import os
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

app = FastAPI(title="Transcriber API")

# Configure CORS - strictly default to FRONTEND_URL environment variable, not "*"
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5000")
origins = [frontend_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Header
from pydantic import BaseModel

# Initialize Groq client
# We create a helper function to get the client, using the user-provided key if present,
# otherwise falling back to the server environment variable.
def get_groq_client(user_key: str = None):
    api_key = user_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API Key is missing. Please provide it in the frontend settings or configure it on the server."
        )
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Groq client: {str(e)}"
        )

async def transcribe_audio(audio_bytes: bytes, user_key: str = None):
    client = get_groq_client(user_key)
    
    # Run the synchronous Groq API call in a thread pool to prevent blocking the async event loop
    def sync_transcribe():
        raw_response = client.audio.transcriptions.with_raw_response.create(
            model="whisper-large-v3-turbo",
            file=("chunk.webm", audio_bytes),  # Explicitly pass chunk.webm filename to inform Groq of the format
            language="en",
            response_format="text"
        )
        headers = raw_response.headers
        text = raw_response.parse()
        
        remaining = headers.get("x-ratelimit-remaining-requests")
        limit = headers.get("x-ratelimit-limit-requests")
        
        return text, remaining, limit
        
    try:
        return await run_in_threadpool(sync_transcribe)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Groq API transcription request failed: {str(e)}"
        )

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), authorization: str = Header(None)):
    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio chunk received.")
        
    user_key = None
    if authorization and authorization.startswith("Bearer "):
        user_key = authorization.replace("Bearer ", "").strip()

    transcribed_text, remaining, limit = await transcribe_audio(audio_bytes, user_key)
    return {
        "text": transcribed_text,
        "remaining_requests": int(remaining) if remaining else None,
        "limit_requests": int(limit) if limit else None
    }

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest, authorization: str = Header(None)):
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text received.")
        
    user_key = None
    if authorization and authorization.startswith("Bearer "):
        user_key = authorization.replace("Bearer ", "").strip()
        
    client = get_groq_client(user_key)
    
    def sync_summarize():
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert transcriber and editor. Your task is to clean up, format, and structure the following raw spoken transcript into a highly readable, detailed document.\n"
                        "CRITICAL REQUIREMENT: Do NOT summarize, condense, or omit any details, arguments, examples, or spoken content. The user needs the full detailed transcript.\n"
                        "FIRST, output exactly one line in this format (no extra text before it):\n"
                        "TITLE: <a concise, specific title for the ENTIRE document, 4-8 words, based on the overall topic discussed, not just the first point made>\n"
                        "Then on the next line, begin the formatted document. Perform the following formatting tasks:\n"
                        "1. Insert logical paragraph breaks and structure the text with clear Markdown headings (e.g. #, ##, ###) based on the topics discussed.\n"
                        "2. Fix grammatical errors, run-on sentences, and remove filler words (like 'um', 'uh', 'like') while keeping the exact meaning and detailed content.\n"
                        "3. Highlight key terms, definitions, or important concepts in bold.\n"
                        "4. Use bullet points or lists where structured lists are spoken.\n"
                        "Ensure the output is clean, professional, and retains 100% of the transcribed details and spoken text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Transcript:\n\n{text}"
                }
            ],
            temperature=0.3
        )
        return completion.choices[0].message.content

    try:
        raw_output = await run_in_threadpool(sync_summarize)
        
        # Split out the TITLE line from the rest of the content
        match = re.match(r"^TITLE:\s*(.+?)\n+(.*)", raw_output, re.DOTALL)
        if match:
            generated_title = match.group(1).strip()
            formatted_content = match.group(2).strip()
        else:
            # Fallback if the model didn't follow the format
            generated_title = "Untitled Note"
            formatted_content = raw_output

        return {
            "title": generated_title,
            "content": formatted_content
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Groq API formatting failed: {str(e)}"
        )

@app.post("/check-quota")
async def check_quota(authorization: str = Header(None)):
    user_key = None
    if authorization and authorization.startswith("Bearer "):
        user_key = authorization.replace("Bearer ", "").strip()
        
    client = get_groq_client(user_key)
    
    # 0.1-second silent mono 16kHz PCM WAV file
    tiny_wav = (
        b'RIFF\xa4\x0c\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00'
        b'\x00}\x00\x00\x02\x00\x10\x00data\x80\x0c\x00\x00' + b'\x00' * 3200
    )
    
    def sync_check():
        raw_response = client.audio.transcriptions.with_raw_response.create(
            model="whisper-large-v3-turbo",
            file=("silence.wav", tiny_wav),
            language="en",
            response_format="text"
        )
        headers = raw_response.headers
        remaining = headers.get("x-ratelimit-remaining-requests")
        limit = headers.get("x-ratelimit-limit-requests")
        return remaining, limit

    try:
        remaining, limit = await run_in_threadpool(sync_check)
        return {
            "remaining_requests": int(remaining) if remaining else None,
            "limit_requests": int(limit) if limit else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check quota: {str(e)}"
        )

@app.get("/health")
def health():
    return {"status": "ok"}

