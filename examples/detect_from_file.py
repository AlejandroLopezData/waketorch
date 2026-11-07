import json

import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL_PATH = "model_v1.onnx"
AUDIO_PATH = "my_audio.wav"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
info = json.loads(session.get_modelmeta().custom_metadata_map["wakeforge"])
input_name = session.get_inputs()[0].name

window = info["window_samples"]
threshold = info["threshold"]
sample_rate = info["input"]["sample_rate"]

audio, rate = sf.read(AUDIO_PATH, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if rate != sample_rate:
    raise SystemExit(f"Resample the audio to {sample_rate} Hz first (it is {rate} Hz).")
audio = np.pad(audio, (max(0, window - len(audio)), 0))

best = 0.0
for end in range(window, len(audio) + 1, int(0.1 * sample_rate)):
    chunk = audio[end - window:end]
    score = float(session.run(None, {input_name: chunk[None, :]})[0][0])
    best = max(best, score)

print(f"best score: {best:.3f} (threshold {threshold:.2f})")
print("WAKE WORD HEARD" if best >= threshold else "no wake word")