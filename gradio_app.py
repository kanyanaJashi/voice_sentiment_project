# gradio_app.py
import gradio as gr
from pipeline import full_pipeline

gr.Interface(
    fn=full_pipeline,
    inputs=gr.Audio(sources="upload", type="filepath", label="Upload a WAV file"),
    outputs=[
        gr.Textbox(label="Raw Transcription"),
        gr.Textbox(label="Corrected Transcript"),
        gr.Label(label="Sentiment"),
        gr.Number(label="Confidence Score")
    ],
    title="📞 JASHI Voice Sentiment Detector",
    description="Detects sentiment in customer voice calls using Wav2Vec2, grammar correction, and BERT."
).launch()
