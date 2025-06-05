from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from dotenv import load_dotenv
import whisper
import tempfile
import json
from typing import List, Dict
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import io
from datetime import datetime
import subprocess
import PyPDF2

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Whisper model
whisper_model = whisper.load_model("small")

# Initialize CLIP model for image understanding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize FAISS index for vector storage
dimension = 512  # CLIP embedding dimension
index = faiss.IndexFlatL2(dimension)

# Define professional behavior indicators
PROFESSIONAL_INDICATORS = [
    "professional attire",
    "good posture",
    "eye contact",
    "confident demeanor",
    "clean background",
    "proper lighting",
    "business casual",
    "well-groomed appearance",
    "professional setting",
    "appropriate facial expressions"
]

# Create embeddings for professional indicators
indicator_embeddings = []
for indicator in PROFESSIONAL_INDICATORS:
    inputs = clip_processor(text=indicator, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    indicator_embeddings.append(text_features.numpy().flatten())

# Add embeddings to FAISS index
indicator_embeddings = np.array(indicator_embeddings)
index.add(indicator_embeddings)

# Define visual cues for body language and confidence
VISUAL_CUES = [
    "person sitting upright",
    "person making eye contact",
    "person smiling",
    "person with open body language",
    "person with confident posture",
    "person with crossed arms",
    "person looking away",
    "person slouching",
    "person frowning"
]

# Create embeddings for visual cues
cue_embeddings = []
for cue in VISUAL_CUES:
    inputs = clip_processor(text=cue, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    cue_embeddings.append(text_features.numpy().flatten())

cue_embeddings = np.array(cue_embeddings)

def extract_frames(video_path, num_frames=5):
    """Extract frames from video using ffmpeg"""
    frames = []
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    try:
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
    except:
        duration = 10  # Default duration if can't get actual duration
    
    for i in range(num_frames):
        timestamp = (duration * i) / num_frames
        output_path = f"{video_path}_frame_{i}.jpg"
        frame_cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        try:
            subprocess.run(frame_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frames.append(output_path)
        except subprocess.CalledProcessError:
            continue
    return frames

def analyze_visual_professionalism(frame_paths):
    """Analyze frames for professional behavior indicators"""
    frame_scores = []
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path)
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            
            # Get similarity scores with professional indicators
            similarities = []
            for indicator_embedding in indicator_embeddings:
                similarity = torch.nn.functional.cosine_similarity(
                    image_features, 
                    torch.tensor(indicator_embedding).unsqueeze(0)
                )
                similarities.append(similarity.item())
            
            frame_scores.append(max(similarities))
        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            continue
    
    return {
        "average_score": sum(frame_scores) / len(frame_scores) if frame_scores else 0,
        "frame_scores": frame_scores
    }

def analyze_visual_cues(frame_paths):
    cue_scores = {cue: [] for cue in VISUAL_CUES}
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path)
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            for i, cue in enumerate(VISUAL_CUES):
                similarity = torch.nn.functional.cosine_similarity(
                    image_features,
                    torch.tensor(cue_embeddings[i]).unsqueeze(0)
                )
                cue_scores[cue].append(similarity.item())
        except Exception as e:
            continue
    # Aggregate scores
    aggregated = {cue: float(np.mean(scores)) if scores else 0.0 for cue, scores in cue_scores.items()}
    return aggregated

@app.get("/api/questions")
async def get_questions():
    return {"questions": PROFESSIONAL_INDICATORS}

@app.post("/api/analyze")
async def analyze_interview(video: UploadFile, resume: UploadFile = None):
    try:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        # Extract audio using ffmpeg
        audio_path = temp_video_path.replace(".webm", ".wav")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as ffmpeg_err:
            raise HTTPException(status_code=500, detail=f"ffmpeg audio extraction failed: {ffmpeg_err.stderr.decode()}")

        # Transcribe audio using Whisper
        transcription = whisper_model.transcribe(audio_path)
        transcript_text = transcription["text"]

        # Extract and analyze frames (more frames for better analysis)
        frame_paths = extract_frames(temp_video_path, num_frames=10)
        visual_cue_scores = analyze_visual_cues(frame_paths)

        # Extract resume text if provided
        resume_text = ""
        if resume is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_resume:
                temp_resume.write(await resume.read())
                temp_resume_path = temp_resume.name
            try:
                with open(temp_resume_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                resume_text = ""
            finally:
                os.unlink(temp_resume_path)

        # Compose prompt for GPT-4
        prompt = f"""
You are an expert technical interviewer. Analyze the following candidate's interview and resume. Provide:
- Scores (0-10) for: Professionalism, Communication, Technical Knowledge, Confidence, Overall Impression
- A paragraph summary/feedback
- Consider the following visual cues (scores are cosine similarities, higher is better):
{json.dumps(visual_cue_scores, indent=2)}

Candidate's Interview Transcript:
{transcript_text}

Resume:
{resume_text}

Respond in JSON with this format:
{{
  "scores": {{
    "professionalism": <int>,
    "communication": <int>,
    "technical_knowledge": <int>,
    "confidence": <int>,
    "overall_impression": <int>
  }},
  "summary": <paragraph summary>
}}
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        import re
        import json as pyjson
        gpt_content = response.choices[0].message.content
        try:
            json_str = re.search(r'\{.*\}', gpt_content, re.DOTALL).group(0)
            gpt_json = pyjson.loads(json_str)
        except Exception:
            gpt_json = {"error": "Could not parse GPT response", "raw": gpt_content}

        # Clean up temporary files
        os.unlink(temp_video_path)
        os.unlink(audio_path)
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.unlink(frame_path)

        # Prepare analysis results
        analysis = {
            "scores": gpt_json.get("scores"),
            "summary": gpt_json.get("summary"),
            "transcription": transcript_text,
            "visual_cues": visual_cue_scores,
            "timestamp": datetime.now().isoformat()
        }

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 