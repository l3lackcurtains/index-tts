import time
import torch
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
        self.available_voices = self._scan_voices()
        
    def _scan_voices(self) -> dict:
        """Scan the voices directory and return available voices"""
        voices = {}
        for voice_file in self.voices_dir.glob("*.mp3"):
            voices[voice_file.stem] = str(voice_file)
        return voices
    
    def get_available_voices(self) -> list:
        """Return list of available voice names"""
        return list(self.available_voices.keys())
        
    def generate(self, voice_name: str, text: str, output_path: str) -> dict:
        """
        Generate audio from text using the specified voice
        
        Args:
            voice_name: Name of the voice to use (must be in available_voices)
            text: Text to convert to speech
            output_path: Where to save the generated audio
        
        Returns:
            dict containing generation stats
        """
        if voice_name not in self.available_voices:
            return {
                "success": False,
                "message": f"Voice '{voice_name}' not found. Available voices: {self.get_available_voices()}",
                "execution_time": None,
                "output_path": None
            }
        
        print(f"Starting audio generation for text: {text[:50]}...")
        start_time = time.time()
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            print("Generating audio with TTS model...")
            # Generate audio
            with torch.inference_mode():
                self.tts.infer(
                    audio_prompt=self.available_voices[voice_name],
                    text=text,
                    output_path=output_path
                )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"Audio generation completed in {execution_time:.2f} seconds")
            print(f"Output saved to: {output_path}")
            
            return {
                "success": True,
                "message": "Audio generated successfully",
                "execution_time": f"{execution_time:.2f}",
                "output_path": output_path
            }
            
        except Exception as e:
            print(f"Error during audio generation: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating audio: {str(e)}",
                "execution_time": None,
                "output_path": None
            }

if __name__ == "__main__":
    print("Initializing TTS Generator...")
    init_start = time.time()
    generator = TTSGenerator()
    init_time = time.time() - init_start
    print(f"Initialization completed in {init_time:.2f} seconds")

    # Test the TTS generator
    test_text = "Hello, this is a test of the TTS system."
    print(f"\nGenerating test audio with text: {test_text}")
    result = generator.generate(
        voice_name="coolio_1",
        text=test_text,
        output_path="output/test.wav"
    )
    
    if result["success"]:
        print(f"\nSummary:")
        print(f"- Total generation time: {result['execution_time']} seconds")
        print(f"- Output file: {result['output_path']}")
    else:
        print(f"\nGeneration failed: {result['message']}")
