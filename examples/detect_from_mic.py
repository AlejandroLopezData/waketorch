import json
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd

MODEL_PATH = "model_v1.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
info = json.loads(session.get_modelmeta().custom_metadata_map["wakeforge"])
input_name = session.get_inputs()[0].name

sample_rate = info["input"]["sample_rate"]
window = info["window_samples"]
threshold = info["threshold"]
step = int(0.1 * sample_rate)

buffer = np.zeros(window, dtype=np.float32)
ready, last_hit = True, 0.0

print("Listening... press Ctrl+C to stop.")
try:
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=step) as mic:
        while True:
            chunk, _ = mic.read(step)
            buffer = np.concatenate([buffer[step:], chunk[:, 0]])
            score = float(session.run(None, {input_name: buffer[None, :]})[0][0])

            if not ready and score < threshold / 2 and time.time() - last_hit > window / sample_rate:
                ready = True
            if ready and score >= threshold:
                ready, last_hit = False, time.time()
                print(f"Wake word detected! (score {score:.2f})")
except KeyboardInterrupt:
    print("Stopped.")