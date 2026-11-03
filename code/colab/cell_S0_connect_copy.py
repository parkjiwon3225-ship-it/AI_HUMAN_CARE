# ===== cell S0 (notebook D, CPU runtime): connect Drive, copy the needed files, install libraries =====
from google.colab import drive
drive.mount('/content/drive')
!pip -q install onnxruntime soundfile
import os, shutil, glob
L = "/content/drive/MyDrive/LAB"
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/pi", exist_ok=True)
os.makedirs(L + "/streams", exist_ok=True)
for p in glob.glob(L + "/processed16k/*Validation*"):          # validation windows + meta only
    shutil.copy(p, "/content/data/")
for fn in ("speech_test.npy", "speech_test.csv"):               # unseen-speaker speech windows
    shutil.copy(L + "/speech16k/" + fn, "/content/data/")
for fn in ("urgent_v1_speech.onnx", "mel_fb.npy", "urgent_v1_speech_config.json"):
    shutil.copy(L + "/models/pi/" + fn, "/content/pi/")
print("data:", sorted(os.listdir("/content/data")))
print("pi  :", sorted(os.listdir("/content/pi")))
