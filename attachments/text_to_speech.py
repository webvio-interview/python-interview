import whisper

def transcriber():
    mp3_filepath = "./your_audio_file.mp3"
    print(f"Loading Whisper model...")
    model = whisper.load_model("base") 
    print(f"Transcribing {mp3_filepath}...")
    result = model.transcribe(mp3_filepath)
    return result["text"]

# Example usage:
transcribed_text = transcriber()
print("\n--- Transcription Result ---")
print(transcribed_text)
