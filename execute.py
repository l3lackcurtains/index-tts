import time
from indextts.infer import IndexTTS

tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "coolio_1.mp3"
text = "(furrowing brow in concern) Sounds rough, friend. Is there anything specific that's got you feeling this way? If it helps, let's talk about some awesome things - like sports or music. What do you enjoy most"

output_path = "gen.wav"

start_time = time.time()
tts.infer(voice, text, output_path)
end_time = time.time()

execution_time = end_time - start_time
print(f"TTS inference completed in {execution_time:.2f} seconds")
