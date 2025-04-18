
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from tts import TTSGenerator

class TTSRequest(BaseModel):
    voice_name: str
    text: str

app = FastAPI(title="TTS API")

# Initialize TTS Generator
tts_generator = TTSGenerator()

# Create directory for outputs
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/voices")
def list_voices():
    """Get list of available voices"""
    return {
        "voices": tts_generator.get_available_voices()
    }

@app.post("/generate")
def generate_speech(request: TTSRequest):
    """
    Generate speech from text using the specified voice
    
    Args:
        request: TTSRequest containing voice_name and text
    """
    try:
        # Generate unique filename for output
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = OUTPUT_DIR / output_filename
        
        # Generate speech
        result = tts_generator.generate(
            voice_name=request.voice_name,
            text=request.text,
            output_path=str(output_path)
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
            
        # Return the generated audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_speech.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5959)

