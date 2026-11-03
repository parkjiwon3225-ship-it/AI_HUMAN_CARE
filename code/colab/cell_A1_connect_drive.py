# ===== cell A1 (notebook A): connect Drive + install soundfile =====
from google.colab import drive
drive.mount('/content/drive')
!pip -q install soundfile
