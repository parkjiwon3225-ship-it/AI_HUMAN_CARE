# ===== cell P0 (notebook C, CPU runtime): connect Drive + install export libraries =====
from google.colab import drive
drive.mount('/content/drive')
!pip -q install onnx onnxruntime
