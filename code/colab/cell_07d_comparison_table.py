# ===== cell 7d: comparison table + breakdown of v1_speech =====
table(RES)
breakdown(lambda w: prob_wavs(get_model("v1_speech"), w), "v1_speech")
