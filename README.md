# AI Video Interview Tool

An MVP for an AI-powered video interviewing tool that analyzes candidate responses using multimodal analysis.

## Features

- Video recording of interview responses
- Speech-to-text transcription using Whisper
- Visual analysis of candidate's presentation
- AI-powered response analysis using GPT-4
- Scoring on multiple criteria including professionalism, communication skills, and more

## Setup

### Prerequisites

- Node.js (v18 or higher)
- Python 3.8 or higher
- OpenAI API key

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:3000

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Start the backend server:
```bash
python main.py
```

The backend will be available at http://localhost:8000

## Usage

1. Open http://localhost:3000 in your browser
2. Click "Start Interview" to begin
3. Allow camera and microphone access when prompted
4. Record your response to each question
5. Upload your resume when prompted
6. Receive AI analysis of your interview performance

## Technical Stack

- Frontend: Next.js, React, Tailwind CSS
- Backend: FastAPI, Python
- AI/ML: OpenAI GPT-4, Whisper, OpenCV
- Video Processing: MoviePy

## Note

This is an MVP version. Some features like advanced visual analysis and resume parsing are simplified for demonstration purposes. 