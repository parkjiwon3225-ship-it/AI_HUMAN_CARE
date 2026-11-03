# ===== cell 1: connect Drive + copy the converted data into the Colab VM =====
from google.colab import drive
drive.mount('/content/drive')
import os, shutil
base = "/content/drive/MyDrive"
for src in [base + "/LAB/processed16k", base + "/processed16k"]:
    if os.path.isdir(src):
        break
else:
    raise FileNotFoundError("processed16k folder not found in Drive")
if not os.path.exists("/content/data/meta_Training_urgent.csv"):
    shutil.copytree(src, "/content/data", dirs_exist_ok=True)
print("data from:", src, "|", len(os.listdir("/content/data")), "files")
