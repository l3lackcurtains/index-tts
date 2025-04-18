
import io
import wave
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from tts import TTSGenerator

class TTSRequest(BaseModel):
    name: str  # Changed from voice_name to name
    text: str

app = FastAPI(title="TTS API")

# Initialize TTS Generator
tts_generator = TTSGenerator()

@app.get("/voices")
def list_voices():
    """Get list of available voices"""
    return {
        "voices": tts_generator.get_available_voices()
    }

@app.post("/generate-audio")  # New endpoint
def generate_audio(request: TTSRequest):
    """
    Generate speech from text using the specified voice
    
    Args:
        request: TTSRequest containing name and text
    """
    try:
        # Generate speech
        result = tts_generator.generate(
            voice_name=request.name,  # Using name from request
            text=request.text
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # sample rate
            wav_file.writeframes(result["audio_data"].tobytes())
            
        # Return the WAV file data
        return Response(
            content=wav_buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{result.get('generation_time', 0):.2f}s"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the old endpoint for backward compatibility
@app.post("/generate")
def generate_speech(request: TTSRequest):
    """Deprecated: Use /generate-audio instead"""
    return generate_audio(request)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5959)

