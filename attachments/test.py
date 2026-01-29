# import whisper

# # Load a model (choose from tiny, base, small, medium, or large)
# model = whisper.load_model("base")

# # Transcribe an audio file
# result = model.transcribe("audio.mp3")



# print(result["text"])



import whisper

model = whisper.load_model("medium")
result = model.transcribe("./audio_file.mp3")
print(result["text"])