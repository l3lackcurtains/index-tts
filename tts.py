import time
import torch
import torchaudio
from pathlib import Path
from indextts.infer import IndexTTS

class TTSGenerator:
    def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", use_fp16=True):
        """Initialize TTS Generator with model settings"""
        self.tts = IndexTTS(
            model_dir=model_dir,
            cfg_path=cfg_path,
            is_fp16=use_fp16
        )
        # Enable CUDA kernel for faster inference
        self.tts.bigvgan.h["use_cuda_kernel"] = True
        
        # Define available voices
        self.voices_dir = Path("voices")
        self.voices_dir.mkdir(exist_ok=True)
        self.voice_cache = {}  # Changed from available_voices to voice_cache
        self._preload_voices()
        print(f"Loaded {len(self.voice_cache)} voices: {list(self.voice_cache.keys())}")
        
    def _preload_voices(self):
        """Pre-load and cache voice files"""
        # Support both mp3 and wav files
        for ext in ["*.mp3", "*.wav"]:
            for voice_file in self.voices_dir.glob(ext):
                voice_name = voice_file.stem
                try:
                    # Load audio and convert to mono if needed
                    audio, sr = torchaudio.load(str(voice_file))
                    if audio.size(0) > 1:  # convert to mono
                        audio = audio.mean(dim=0, keepdim=True)
                    
                    # Resample to 24kHz if needed
                    if sr != 24000:
                        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
                    
                    # Store the file path in the cache
                    self.voice_cache[voice_name] = str(voice_file)
                    print(f"Loaded voice: {voice_name}")
                except Exception as e:
                    print(f"Failed to load voice {voice_name}: {e}")
    
    def get_available_voices(self) -> list:
        """Return list of available voice names"""
        return list(self.voice_cache.keys())
        
    def generate(self, voice_name: str, text: str):
        """
        Generate audio from text using the specified voice
        
        Args:
            voice_name: Name of the voice to use (must be in voice_cache)
            text: Text to convert to speech
        """
        if voice_name not in self.voice_cache:
            return {
                "success": False,
                "message": f"Voice not found. Available: {list(self.voice_cache.keys())}"
            }
        
        try:
            start_time = time.time()
            
            # Get voice file path from cache
            audio_prompt = self.voice_cache[voice_name]
            
            # Split long text for faster processing
            max_length = 200
            text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            wavs = []
            for chunk in text_chunks:
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    wav = self.tts.infer_return_wav(
                        audio_prompt=audio_prompt,
                        text=chunk
                    )
                    wavs.append(wav)
            
            # Combine chunks
            if len(wavs) > 1:
                wav = torch.cat(wavs, dim=1)
            else:
                wav = wavs[0]
                
            generation_time = time.time() - start_time
            print(f"Generated audio for voice '{voice_name}' in {generation_time:.2f} seconds")
            
            return {
                "success": True,
                "audio_data": wav.cpu().numpy(),
                "generation_time": generation_time
            }
            
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

