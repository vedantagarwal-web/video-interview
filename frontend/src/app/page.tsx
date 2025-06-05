'use client';

import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const startInterview = async () => {
    setInterviewStarted(true);
    setCurrentQuestion("Tell me about yourself and your experience.");
  };

  const startRecording = () => {
    setRecordedChunks([]);
    if (webcamRef.current && webcamRef.current.stream) {
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream);
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data]);
        }
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleResumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setResumeFile(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    setAnalysis(null);
    try {
      // Combine recorded chunks into a single blob
      const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
      const formData = new FormData();
      formData.append('video', videoBlob, 'interview.webm');
      if (resumeFile) {
        formData.append('resume', resumeFile);
      }
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAnalysis(response.data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'An error occurred during analysis.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen p-8 bg-gray-100">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">AI Video Interview</h1>
        {!interviewStarted ? (
          <div className="text-center">
            <button
              onClick={startInterview}
              className="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition"
            >
              Start Interview
            </button>
          </div>
        ) : analysis ? (
          <div className="bg-white p-6 rounded-lg shadow-md mt-8">
            <h2 className="text-2xl font-semibold mb-4">Interview Analysis</h2>
            {analysis.scores && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Scores</h3>
                <ul className="mb-2">
                  {Object.entries(analysis.scores).map(([key, value]) => (
                    <li key={key} className="capitalize flex justify-between">
                      <span>{key.replace(/_/g, ' ')}:</span>
                      <span className="font-mono">{value}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysis.summary && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Summary</h3>
                <p className="text-gray-700 whitespace-pre-line">{analysis.summary}</p>
              </div>
            )}
            {analysis.visual_cues && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Visual Cues Analysis</h3>
                <ul>
                  {Object.entries(analysis.visual_cues).reduce<React.ReactNode[]>((acc, [cue, score]) => {
                    if (typeof cue !== 'string') return acc;
                    const scoreNum = typeof score === 'number' ? score : parseFloat(String(score));
                    if (isNaN(scoreNum)) return acc;
                    acc.push(
                      <li key={cue} className="flex justify-between items-center mb-1">
                        <span className="capitalize">{cue.replace(/_/g, ' ')}</span>
                        <span className="ml-2 font-mono">{(scoreNum * 100).toFixed(1)}%</span>
                        <div className="ml-2 w-40 bg-gray-200 rounded h-2">
                          <div
                            className="bg-blue-500 h-2 rounded"
                            style={{ width: `${Math.max(0, Math.min(1, scoreNum)) * 100}%` }}
                          ></div>
                        </div>
                      </li>
                    );
                    return acc;
                  }, [])}
                </ul>
              </div>
            )}
            {analysis.transcription && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Transcription</h3>
                <p className="text-gray-700 whitespace-pre-line">{analysis.transcription}</p>
              </div>
            )}
            <button
              className="mt-4 bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
              onClick={() => window.location.reload()}
            >
              Start Over
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Current Question:</h2>
              <p className="text-gray-700">{currentQuestion}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <Webcam
                ref={webcamRef}
                audio={true}
                className="w-full rounded-lg"
              />
            </div>
            <div className="flex justify-center space-x-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="bg-red-500 text-white px-6 py-3 rounded-lg hover:bg-red-600 transition"
                  disabled={isSubmitting}
                >
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition"
                  disabled={isSubmitting}
                >
                  Stop Recording
                </button>
              )}
            </div>
            {recordedChunks.length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow-md mt-4">
                <h3 className="text-lg font-semibold mb-2">Upload Resume (PDF):</h3>
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={handleResumeChange}
                  className="mb-4"
                  disabled={isSubmitting}
                />
                <button
                  onClick={handleSubmit}
                  className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 disabled:opacity-50"
                  disabled={!resumeFile || isSubmitting}
                >
                  {isSubmitting ? 'Analyzing...' : 'Submit for Analysis'}
                </button>
                {error && <p className="text-red-500 mt-2">{error}</p>}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
} 