import torch
import soundfile as sf
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline
)
from torchaudio.transforms import Resample

# Auto-select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models on appropriate device
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Hugging Face pipelines handle device internally
hf_device = 0 if device == "cuda" else -1
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=hf_device)
grammar_pipeline = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction", device=hf_device)

def transcribe(audio_path):
    waveform_np, sample_rate = sf.read(audio_path)
    if waveform_np.ndim > 1:
        waveform_np = np.mean(waveform_np, axis=1)

    waveform = torch.tensor(waveform_np, dtype=torch.float32)

    if torch.max(torch.abs(waveform)) > 1.0:
        waveform = waveform / torch.max(torch.abs(waveform))

    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def correct_grammar(text):
    return grammar_pipeline(text)[0]['generated_text']

def analyze_sentiment(text):
    return sentiment_pipeline(text)[0]

def full_pipeline(audio_path):
    try:
        transcription = transcribe(audio_path)
        corrected = correct_grammar(transcription)
        sentiment = analyze_sentiment(corrected)
        label = sentiment['label']
        score = round(sentiment['score'], 2)
        return transcription, corrected, label, score
    except Exception as e:
        print(f"Pipeline error: {e}")
        return "Transcription error", "Grammar error", "N/A", 0.0
