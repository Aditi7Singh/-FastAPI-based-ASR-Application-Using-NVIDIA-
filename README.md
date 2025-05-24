#  FastAPI-based ASR Application Using NVIDIA 
Deployed NVIDIA NeMo ASR model optimized with ONNX in a FastAPI app accepting 5–10 sec, 16kHz .wav files at /transcribe. Inference is synchronous due to runtime limits but wrapped in async API. Containerized with a slim Python image. Challenges: ONNX compatibility and audio preprocessing. Future work: async inference and broader audio support.
