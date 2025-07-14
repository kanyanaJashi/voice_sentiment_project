# api_server.py
from fastapi import FastAPI, UploadFile, File
from pipeline import full_pipeline

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    audio_path = "/tmp/" + file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    raw, corrected, label, score = full_pipeline(audio_path)
    return {
        "transcription": raw,
        "corrected": corrected,
        "sentiment": label,
        "confidence": score
    }
