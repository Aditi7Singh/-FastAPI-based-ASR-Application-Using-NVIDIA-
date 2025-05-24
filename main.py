import os
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import librosa
import numpy as np
from pydantic import BaseModel
import torch
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeMo ASR API",
    description="Automatic Speech Recognition API using NVIDIA NeMo Conformer CTC model",
    version="1.0.0"
)

class TranscriptionResponse(BaseModel):
    transcription: str
    duration: float
    sample_rate: int

class ASRModel:
    def __init__(self, model_path: str):
        """Initialize the ONNX ASR model"""
        self.model_path = model_path
        self.session = None
        self.vocab = None
        self.sample_rate = 16000
        self.load_model()
        
    def load_model(self):
        """Load the ONNX model and vocabulary"""
        try:
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Load vocabulary (character-based for CTC)
            self.vocab = self._load_vocabulary()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Available providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_vocabulary(self):
        """Load the Hindi character vocabulary for CTC decoding"""
        # Standard Hindi CTC vocabulary (simplified for demonstration)
        # In a real implementation, this should be loaded from the model's metadata
        vocab = [
            '<blank>', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
            'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ',
            'ं', 'ः', '्', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        return {i: char for i, char in enumerate(vocab)}
    
    def preprocess_audio(self, audio_path: str):
        """Preprocess audio for model input"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Validate duration (5-10 seconds)
            duration = len(audio) / sr
            if duration < 5 or duration > 10:
                raise ValueError(f"Audio duration {duration:.2f}s is outside the 5-10 second range")
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Convert to mel-spectrogram or raw audio based on model requirements
            # For this example, we'll use raw audio features
            audio_features = self._extract_features(audio)
            
            return audio_features, duration
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def _extract_features(self, audio):
        """Extract features from raw audio"""
        # For NeMo Conformer models, typically we need mel-spectrogram features
        # This is a simplified version - in practice, you'd use NeMo's feature extraction
        
        # Compute mel-spectrogram
        n_fft = 512
        hop_length = 160
        n_mels = 80
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, features) format
        log_mel = log_mel.T
        
        # Add batch dimension
        features = np.expand_dims(log_mel, axis=0).astype(np.float32)
        
        return features
    
    def ctc_decode(self, logits):
        """Simple CTC decoding"""
        # Get the most likely character at each time step
        predictions = np.argmax(logits, axis=-1)
        
        # Remove consecutive duplicates and blank tokens
        decoded = []
        prev_char = None
        
        for pred in predictions[0]:  # Remove batch dimension
            if pred != 0 and pred != prev_char:  # 0 is blank token
                if pred < len(self.vocab):
                    decoded.append(self.vocab[pred])
            prev_char = pred
        
        return ''.join(decoded)
    
    async def transcribe(self, audio_path: str):
        """Transcribe audio file"""
        try:
            # Preprocess audio
            features, duration = self.preprocess_audio(audio_path)
            
            # Create input dictionary for ONNX model
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            loop = asyncio.get_event_loop()
            logits = await loop.run_in_executor(
                None, 
                lambda: self.session.run(None, {input_name: features})
            )
            
            # Decode the output
            transcription = self.ctc_decode(logits[0])
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

# Global model instance
asr_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global asr_model
    model_path = os.getenv("MODEL_PATH", "models/stt_hi_conformer_ctc_medium.onnx")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    asr_model = ASRModel(model_path)
    logger.info("ASR model initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "NeMo ASR API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": asr_model is not None,
        "supported_formats": [".wav"],
        "max_duration": "10 seconds",
        "min_duration": "5 seconds"
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file
    
    - **file**: Audio file (.wav format, 5-10 seconds, 16kHz sample rate)
    
    Returns the transcribed text along with audio metadata
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only .wav files are supported"
        )
    
    # Validate file size (rough check for duration)
    if file.size > 10 * 16000 * 2:  # 10 seconds * 16kHz * 2 bytes per sample
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum duration is 10 seconds."
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe the audio
        transcription, duration = await asr_model.transcribe(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return TranscriptionResponse(
            transcription=transcription,
            duration=duration,
            sample_rate=asr_model.sample_rate
        )
        
    except ValueError as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during transcription")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
