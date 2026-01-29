import whisper
import os


def transcriber():
model = whisper.load_model("base")


audio_file_path = "audio_file.mp3"

# Transcribe the audio
result = model.transcribe(audio_file_path)

# Print or save the transcription
print(result["text"])

# Optional: Save the transcription to a text file
with open("transcription_output.txt", "w") as file:
    file.write(result["text"])