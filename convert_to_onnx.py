#!/usr/bin/env python3
"""
Script to convert NVIDIA NeMo ASR model to ONNX format
"""

import os
import argparse
import logging
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_convert_model(model_name: str, output_dir: str):
    """
    Download NeMo model and convert to ONNX
    
    Args:
        model_name: Name of the NeMo model
        output_dir: Directory to save ONNX model
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the pre-trained model
        logger.info(f"Downloading model: {model_name}")
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)
        
        # Set model to evaluation mode
        asr_model.eval()
        
        # Get model configuration
        cfg = asr_model.cfg
        logger.info(f"Model config: {OmegaConf.to_yaml(cfg)}")
        
        # Create dummy input for ONNX export
        # NeMo models typically expect processed features, not raw audio
        batch_size = 1
        seq_len = 200  # Typical sequence length for 5-10 second audio
        feat_dim = cfg.preprocessor.features  # Usually 80 for mel-spectrogram
        
        dummy_input = torch.randn(batch_size, seq_len, feat_dim)
        dummy_input_length = torch.tensor([seq_len], dtype=torch.long)
        
        # ONNX export path
        onnx_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.onnx")
        
        logger.info(f"Converting to ONNX format...")
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Output path: {onnx_path}")
        
        # Export to ONNX
        torch.onnx.export(
            asr_model,
            (dummy_input, dummy_input_length),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio_signal', 'length'],
            output_names=['logits'],
            dynamic_axes={
                'audio_signal': {0: 'batch_size', 1: 'time'},
                'length': {0: 'batch_size'},
                'logits': {0: 'batch_size', 1: 'time'}
            }
        )
        
        logger.info(f"ONNX model saved to: {onnx_path}")
        
        # Save vocabulary and model metadata
        vocab_path = os.path.join(output_dir, "vocabulary.txt")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for i, token in enumerate(asr_model.decoder.vocabulary):
                f.write(f"{i}\t{token}\n")
        
        logger.info(f"Vocabulary saved to: {vocab_path}")
        
        # Save model configuration
        config_path = os.path.join(output_dir, "model_config.yaml")
        OmegaConf.save(cfg, config_path)
        logger.info(f"Model config saved to: {config_path}")
        
        return onnx_path
        
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        raise

def verify_onnx_model(onnx_path: str):
    """Verify the converted ONNX model"""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model is valid")
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        logger.info(f"ONNX Runtime providers: {session.get_providers()}")
        
        # Get input/output info
        input_info = [(input.name, input.shape, input.type) for input in session.get_inputs()]
        output_info = [(output.name, output.shape, output.type) for output in session.get_outputs()]
        
        logger.info(f"Model inputs: {input_info}")
        logger.info(f"Model outputs: {output_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert NeMo ASR model to ONNX")
    parser.add_argument(
        "--model",
        default="nvidia/stt_hi_conformer_ctc_medium",
        help="NeMo model name"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the converted ONNX model"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert model
        onnx_path = download_and_convert_model(args.model, args.output_dir)
        
        # Verify if requested
        if args.verify:
            verify_onnx_model(onnx_path)
        
        logger.info("Model conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()