import sounddevice as sd
import numpy as np
import whisper
import tempfile
import queue
import sys
import os
import soundfile as sf

# Load Whisper model
model = whisper.load_model("small")  # tiny/base/medium/large

# Queue for audio chunks
q = queue.Queue()
all_audio = []  # to store all chunks for final transcription

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    all_audio.append(indata.copy())  # store chunk for final transcription

# Parameters
samplerate = 16000  # Whisper prefers 16kHz
blocksize = 3       # seconds per chunk
frames_per_block = samplerate * blocksize

print("🎤 Listening... Press Ctrl+C to stop.")

try:
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=frames_per_block):
        while True:
            audio = q.get()

            # Make sure audio is 2D: (samples, channels)
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
                sf.write(wav_path, audio, samplerate)

            # Transcribe chunk
            result = model.transcribe(wav_path, fp16=False)
            print("⏱ Chunk:", result["text"].strip())

            os.remove(wav_path)

except KeyboardInterrupt:
    print("\n🛑 Stopped recording.")

    # Combine all audio chunks into one big array
    full_audio = np.concatenate(all_audio, axis=0)

    # Save full session to a temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        sf.write(wav_path, full_audio, samplerate)

    # Final full transcription
    print("\n📝 Transcribing entire session...")
    result = model.transcribe(wav_path, fp16=False)
    print("\n=== FULL TRANSCRIPTION ===\n")
    print(result["text"].strip())

    os.remove(wav_path)
    print("\n✅ Done.")
