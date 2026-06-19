import os
import re
import httpx
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
                        "You are an expert content editor who turns raw spoken transcripts into a polished, "
                        "detailed study summary — not a verbatim cleanup, and not a shallow summary either. "
                        "Aim for the depth a top student's notes would have: every important point, argument, "
                        "and example preserved, but rewritten in clear, professional language, with filler and "
                        "repetition removed.\n\n"
                        "Output in exactly this structure:\n\n"
                        "TITLE: <a concise, specific title for the ENTIRE document, 4-8 words, based on the overall topic>\n"
                        "TLDR:\n"
                        "- <first essential takeaway, one sentence>\n"
                        "- <second essential takeaway, one sentence>\n"
                        "- <third essential takeaway, one sentence>\n"
                        "---\n"
                        "<then the full formatted document>\n\n"
                        "Rules for the full document:\n"
                        "1. STRUCTURE: Use Markdown headings (#, ##, ###) to organize by topic.\n"
                        "2. LANGUAGE: Rewrite spoken sentences into clean, well-formed prose. Do not just trim filler words — "
                        "actually restate ideas more clearly and concisely where the original phrasing was rambling or repetitive.\n"
                        "3. EMPHASIS: Bold key terms, names, and concepts.\n"
                        "4. LISTS: Use bullet points for enumerated items.\n"
                        "5. CODE & FORMULAS: If the topic involves code, technical steps, mathematical formulas, or "
                        "structured data, include them in fenced code blocks using triple backticks, even if the speaker "
                        "only described them verbally rather than writing them out. Reconstruct them accurately based on context.\n"
                        "6. EXTRA HELPFUL CONTEXT: Where it adds genuine value, supplement the transcript with brief "
                        "relevant context the speaker didn't explicitly say — but keep this additive, never contradicting "
                        "or replacing what was actually said.\n"
                        "7. HIGHLIGHT BOXES: Use Markdown blockquote syntax (lines starting with '>') for the most important "
                        "points. Use ONE of these four labels depending on what kind of point it is:\n"
                        "   > **Key Insight:** <an important realization or core idea>\n"
                        "   > **Definition:** <a technical term or concept being explained>\n"
                        "   > **Common Mistake:** <a pitfall or warning the speaker mentioned>\n"
                        "   > **Action Item:** <something concrete the reader/listener should do>\n"
                        "   Use these sparingly — 3-6 total across the document, only for genuinely important points.\n"
                        "8. OMIT IRRELEVANT CONTENT: Leave out jokes, banter, off-topic tangents, small talk, "
                        "filler anecdotes, sponsor reads/ads, and any other content that doesn't contribute to the "
                        "actual subject matter.\n\n"
                        "Do not omit substantive, on-topic content — but you SHOULD condense redundant or repetitive "
                        "spoken passages into tighter, clearer writing, and you SHOULD fully drop the off-topic material "
                        "described in rule 8."
                    )
                },
                {
                    "role": "user",
                    "content": f"Transcript:\n\n{text}"
                }
            ],
            temperature=0.4
        )
        return completion.choices[0].message.content

    try:
        raw_output = await run_in_threadpool(sync_summarize)
        
        # --- Parse TITLE ---
        title_match = re.search(r"^TITLE:\s*(.+)$", raw_output, re.MULTILINE)
        generated_title = title_match.group(1).strip() if title_match else "Untitled Note"

        # --- Parse TLDR bullets (between "TLDR:" and the "---" delimiter) ---
        tldr_match = re.search(r"TLDR:\s*\n(.*?)\n---\s*\n", raw_output, re.DOTALL)
        tldr_bullets = []
        if tldr_match:
            for line in tldr_match.group(1).split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    tldr_bullets.append(line[2:].strip())

        # --- Everything after the "---" delimiter is the main content ---
        content_match = re.search(r"\n---\s*\n(.*)", raw_output, re.DOTALL)
        formatted_content = content_match.group(1).strip() if content_match else raw_output

        # Fallback: if parsing failed entirely, just use the raw output as content
        if not title_match and not tldr_match and not content_match:
            formatted_content = raw_output

        # --- Second LLM Call to condense to 1-page ---
        def sync_condense():
            one_page_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are condensing a detailed document into a single-page executive summary "
                            "(target: 300-450 words total, must fit comfortably on one printed page).\n\n"
                            "Rules:\n"
                            "1. Use Markdown. Start with one or two short paragraphs covering the core topic, "
                            "then a '## Key Points' section with 4-7 bullet points covering the most important "
                            "specifics (numbers, names, decisions, conclusions).\n"
                            "2. Bold the most important terms.\n"
                            "3. Include AT MOST one highlight box using this format for the single most important "
                            "takeaway in the whole document:\n"
                            "   > **Key Insight:** <one sentence>\n"
                            "4. Do NOT include code blocks, multiple sub-headings, or any content that won't fit "
                            "on one page. Be ruthless about cutting detail — this is a skim-read summary, not the "
                            "full document.\n"
                            "5. Do not add a title line — just start directly with the content."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Detailed document to condense:\n\n{formatted_content}"
                    }
                ],
                temperature=0.4
            )
            return one_page_completion.choices[0].message.content.strip()

        one_page_summary = await run_in_threadpool(sync_condense)

        return {
            "title": generated_title,
            "tldr": tldr_bullets,
            "content": formatted_content,
            "one_page_summary": one_page_summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Groq API formatting failed: {str(e)}"
        )

@app.post("/check-quota")
async def check_quota(authorization: str = Header(None)):
    """
    Check remaining Groq quota without burning any audio credits.
    We hit the cheap /openai/v1/models endpoint which returns the same
    x-ratelimit-* response headers as transcription calls.
    """
    user_key = None
    if authorization and authorization.startswith("Bearer "):
        user_key = authorization.replace("Bearer ", "").strip()

    api_key = user_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API Key is missing."
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Groq API error: {response.text}"
                )
            headers = response.headers
            remaining = headers.get("x-ratelimit-remaining-requests")
            limit = headers.get("x-ratelimit-limit-requests")
            remaining_seconds = headers.get("x-ratelimit-remaining-tokens")  # audio seconds for Whisper
            limit_seconds = headers.get("x-ratelimit-limit-tokens")
            return {
                "remaining_requests": int(remaining) if remaining else None,
                "limit_requests": int(limit) if limit else None,
                "remaining_seconds": int(remaining_seconds) if remaining_seconds else None,
                "limit_seconds": int(limit_seconds) if limit_seconds else None,
            }
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Groq API: {str(e)}"
        )

@app.get("/health")
def health():
    return {"status": "ok"}

