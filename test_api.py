# test_client_roberta.py
import requests

# Endpoint RoBERTa
url = "http://localhost:8000/analyze"
audio_path = "wav_rec_files/Recording 7.wav"

with open(audio_path, "rb") as f:
    files = {"file": (audio_path, f, "audio/wav")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print("🟥 RoBERTa Pipeline Results:")
    print("Transcription :", data["transcription"])
    print("Texte corrigé :", data["corrected"])
    print("Sentiment     :", data["sentiment"])
    print("Confiance     :", data["confidence"])
else:
    print("❌ Erreur RoBERTa:", response.text)
# test_client_roberta.py
