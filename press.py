import sounddevice as sd
import numpy as np
import whisper
import soundfile as sf
import tempfile
import keyboard  # pip install keyboard

# Load Whisper model
model = whisper.load_model("small")

samplerate = 16000  # Whisper prefers 16kHz
channels = 1

print("🎤 Hold SPACE to record. Release to stop and transcribe.")

while True:
    keyboard.wait("space")  # Wait until space is pressed
    print("🎙 Recording... (release SPACE to stop)")

    recording = []  # store audio chunks

    # Record while key is held down
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32") as stream:
        while keyboard.is_pressed("space"):
            audio = stream.read(1024)[0]  # read 1024 frames
            recording.append(audio.copy())

    print("⏹ Stopped recording. Transcribing...")

    # Combine all chunks into one array
    full_audio = np.concatenate(recording, axis=0)

    # Save temporary WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        sf.write(wav_path, full_audio, samplerate)

    # Transcribe
    result = model.transcribe(wav_path, fp16=False)
    print("\n=== TRANSCRIPTION ===")
    print(result["text"].strip())
    print("=====================\n")
