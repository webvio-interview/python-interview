import os
import json
import wave
from fastapi import FastAPI, UploadFile, File, HTTPException
from vosk import Model, KaldiRecognizer, SpkModel
from pydub import AudioSegment

app = FastAPI()

# Load models locally (Ensure these folders exist)
MODEL_PATH = "model"  
SPK_MODEL_PATH = "model-spk" # Optional: Speaker identification model folder

if not os.path.exists(MODEL_PATH):
    raise Exception(f"ASR Model not found at {MODEL_PATH}")

model = Model(MODEL_PATH)
spk_model = SpkModel(SPK_MODEL_PATH) if os.path.exists(SPK_MODEL_PATH) else None

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Convert audio to 16kHz mono WAV (Vosk requirement)
    temp_raw = f"temp_{file.filename}"
    temp_wav = "processed.wav"
    
    with open(temp_raw, "wb") as buffer:
        buffer.write(await file.read())
    
    audio = AudioSegment.from_file(temp_raw)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(temp_wav, format="wav")

    wf = wave.open(temp_wav, "rb")
    rec = KaldiRecognizer(model, wf.getparams().framerate)
    if spk_model:
        rec.SetSpkModel(spk_model) # Enable speaker identification
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    
    results.append(json.loads(rec.FinalResult()))
    
    # 3. Cleanup
    # wf.close()
    # os.remove(temp_raw)
    # os.remove(temp_wav)

    return {"filename": file.filename, "data": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)