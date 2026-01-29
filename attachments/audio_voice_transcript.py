import uuid
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from pyannote.audio import Pipeline
import whisper

app = FastAPI()

# Load models once at startup
whisper_model = whisper.load_model("base")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HF_TOKEN")
)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        audio_path = temp_audio.name

    # Transcription
    transcription = whisper_model.transcribe(audio_path)
    language = transcription["language"]

    # Speaker diarization
    diarization = diarization_pipeline(audio_path)

    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Find text that overlaps this speaker segment
        text_chunks = [
            seg["text"]
            for seg in transcription["segments"]
            if seg["start"] < turn.end and seg["end"] > turn.start
        ]

        if text_chunks:
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "text": " ".join(text_chunks).strip()
            })

    os.remove(audio_path)

    return {
        "job_id": job_id,
        "language": language,
        "segments": segments
    }
