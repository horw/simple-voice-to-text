import sounddevice as sd
import numpy as np
import whisper
import soundfile as sf
import tempfile
from pynput import keyboard  # pip install pynput
import pyperclip  # pip install pyperclip

# Load Whisper model
model = whisper.load_model("small")

samplerate = 16000
channels = 1
recording = []
is_recording = False
min_duration_sec = 2.0  # 🔹 Only transcribe if > 2 seconds

def on_press(key):
    global is_recording, recording
    if key == keyboard.Key.space and not is_recording:
        print("🎙 Recording... (release SPACE to stop)")
        is_recording = True
        recording = []

        def callback(indata, frames, time, status):
            if is_recording:
                recording.append(indata.copy())

        # Start input stream
        listener.stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=callback)
        listener.stream.start()

def on_release(key):
    global is_recording
    if key == keyboard.Key.space and is_recording:
        print("⏹ Stopped recording.")
        is_recording = False
        listener.stream.stop()
        listener.stream.close()

        if recording:
            full_audio = np.concatenate(recording, axis=0)
            duration = len(full_audio) / samplerate  # 🔹 seconds

            if duration >= min_duration_sec:
                print(f"⏳ Audio length: {duration:.2f}s → Transcribing...")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                    sf.write(wav_path, full_audio, samplerate)

                result = model.transcribe(wav_path, fp16=False)
                text = result["text"].strip()

                # 🔹 Copy to clipboard
                pyperclip.copy(text)

                print("\n=== TRANSCRIPTION ===")
                print(text)
                print("=====================\n")
                print("📋 Copied to clipboard!")
            else:
                print(f"⚠️ Audio too short ({duration:.2f}s). Ignored.")

    if key == keyboard.Key.esc:
        print("\n👋 Exiting...")
        return False
    return None


print("🎤 Hold SPACE to record. Release to stop. Press ESC to quit.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
