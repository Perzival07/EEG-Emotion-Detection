#!/usr/bin/env python3
"""
Inference script for Brain EEG emotion classification using trained OpenTSLMFlamingo model.
This script loads a single EEG file and predicts the emotion using the trained model.

Usage:
    python inference_single_eeg.py --eeg_file /path/to/eeg.csv
    python inference_single_eeg.py --eeg_file /path/to/eeg.csv --max_tokens 100
"""

import sys
import os
import argparse
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from time_series_datasets.brain_eeg.brain_eeg_loader import (
    load_eeg_from_csv,
    EMOTION_LABELS,
    EEG_CHANNELS
)
from prompt.text_time_series_prompt import TextTimeSeriesPrompt


# Channel descriptions for prompts (same as in BrainEEGQADataset)
CHANNEL_DESCRIPTIONS = [
    "The following is the EEG signal from the AF3 electrode (left frontal area)",
    "The following is the EEG signal from the F7 electrode (left frontal-temporal area)",
    "The following is the EEG signal from the F3 electrode (left frontal area)",
    "The following is the EEG signal from the FC5 electrode (left frontal-central area)",
    "The following is the EEG signal from the T7 electrode (left temporal area)",
    "The following is the EEG signal from the P7 electrode (left parietal area)",
    "The following is the EEG signal from the O1 electrode (left occipital area)",
    "The following is the EEG signal from the O2 electrode (right occipital area)",
    "The following is the EEG signal from the P8 electrode (right parietal area)",
    "The following is the EEG signal from the T8 electrode (right temporal area)",
    "The following is the EEG signal from the FC6 electrode (right frontal-central area)",
    "The following is the EEG signal from the F4 electrode (right frontal area)",
    "The following is the EEG signal from the F8 electrode (right frontal-temporal area)",
    "The following is the EEG signal from the AF4 electrode (right frontal area)",
]


def create_eeg_batch(eeg_data: np.ndarray, patch_size: int = 4) -> list:
    """
    Create a batch in the format expected by OpenTSLMFlamingo.
    
    Args:
        eeg_data: numpy array of shape (n_channels, n_timepoints)
        patch_size: patch size for time series encoding
        
    Returns:
        List containing a single sample dictionary
    """
    n_channels, n_timepoints = eeg_data.shape
    
    # Convert to tensor
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    
    # Normalize each channel independently (same as training)
    means = eeg_tensor.mean(dim=1, keepdim=True)  # (n_channels, 1)
    stds = eeg_tensor.std(dim=1, keepdim=True)    # (n_channels, 1)
    
    # Handle zero or very small standard deviations
    min_std = 1e-6
    stds = torch.clamp(stds, min=min_std)
    
    eeg_normalized = (eeg_tensor - means) / stds
    
    # Check for NaN/Inf
    if torch.isnan(eeg_normalized).any() or torch.isinf(eeg_normalized).any():
        print(f"Warning: NaN/Inf detected in EEG data, replacing with zeros")
        eeg_normalized = torch.nan_to_num(eeg_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create pre-prompt (same as training)
    pre_prompt = """
You are given EEG (electroencephalogram) data from 14 electrodes placed on the scalp. 
Your task is to classify the emotional state based on analysis of the brain activity patterns.

The EEG data represents electrical activity from different brain regions:
- Frontal electrodes (AF3, AF4, F3, F4, F7, F8): Associated with cognitive processing and emotion regulation
- Central electrodes (FC5, FC6): Bridge between frontal and central regions
- Temporal electrodes (T7, T8): Associated with auditory processing and memory
- Parietal electrodes (P7, P8): Associated with sensory processing
- Occipital electrodes (O1, O2): Associated with visual processing

Possible emotion labels are:
angry, baseline, frightened, happy, protected, sad, satisfied, surprise, unconcerned.

Analyze the patterns in the EEG signals and classify the emotional state.
""".strip()
    
    # Create TextTimeSeriesPrompt objects for each channel
    time_series_prompts = []
    time_series_tensors = []
    time_series_texts = []
    
    for i, (channel_name, channel_desc) in enumerate(zip(EEG_CHANNELS, CHANNEL_DESCRIPTIONS)):
        if i >= n_channels:
            break
        
        # Get normalized channel data as tensor (1D)
        channel_tensor = eeg_normalized[i]  # Shape: (n_timepoints,)
        mean_val = means[i].item()
        std_val = stds[i].item()
        
        # Create text prompt with statistics (same as training)
        text_prompt = f"{channel_desc}, it has mean {mean_val:.4f} and std {std_val:.4f}:"
        
        # Store tensors and texts separately
        time_series_tensors.append(channel_tensor)
        time_series_texts.append(text_prompt)
    
    # Stack all channel tensors into a single tensor (n_channels, n_timepoints)
    # This is what the model expects in pad_and_apply_batch
    stacked_time_series = torch.stack(time_series_tensors)
    
    # Create post-prompt (same as training)
    post_prompt = "Emotion classification:"
    
    # Create sample dictionary (matching PromptWithAnswer.to_dict() format)
    sample = {
        "pre_prompt": pre_prompt,
        "time_series_text": time_series_texts,
        "time_series": stacked_time_series,  # Now a single stacked tensor
        "post_prompt": post_prompt,
        "answer": "",  # Empty for inference
    }
    
    return [sample]


def load_model(checkpoint_path: str, device: str = "cuda") -> OpenTSLMFlamingo:
    """
    Load the trained OpenTSLMFlamingo model from checkpoint.
    
    Args:
        checkpoint_path: Path to the best_model.pt checkpoint
        device: Device to use ('cuda', 'cpu', or 'mps')
        
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Initialize model
    model = OpenTSLMFlamingo(
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
        llm_id="google/gemma-3-270m",
        device=device,
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model_state = checkpoint["model_state"]
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    
    if missing_keys:
        print(f"⚠️  Warning: Missing keys when loading checkpoint:")
        for key in missing_keys[:5]:
            print(f"   - {key}")
        if len(missing_keys) > 5:
            print(f"   ... and {len(missing_keys) - 5} more keys")
    
    if unexpected_keys:
        print(f"⚠️  Warning: Unexpected keys when loading checkpoint:")
        for key in unexpected_keys[:5]:
            print(f"   - {key}")
        if len(unexpected_keys) > 5:
            print(f"   ... and {len(unexpected_keys) - 5} more keys")
    
    # Set to evaluation mode
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    return model


def predict_emotion(
    model: OpenTSLMFlamingo,
    eeg_file_path: str,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> dict:
    """
    Predict emotion from an EEG file.
    
    Args:
        model: Trained OpenTSLMFlamingo model
        eeg_file_path: Path to the EEG CSV file
        max_new_tokens: Maximum number of tokens to generate
        device: Device to use
        
    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*60}")
    print(f"Processing EEG file: {eeg_file_path}")
    print(f"{'='*60}")
    
    # Load EEG data
    eeg_data = load_eeg_from_csv(eeg_file_path, apply_center_crop=True)
    print(f"EEG data shape: {eeg_data.shape} (channels x timepoints)")
    
    # Get ground truth label from file path
    file_name = os.path.basename(eeg_file_path)
    folder_name = os.path.basename(os.path.dirname(eeg_file_path))
    ground_truth = folder_name.lower() if folder_name in [e for e in EMOTION_LABELS] else "unknown"
    
    # Create batch
    batch = create_eeg_batch(eeg_data)
    
    # Run inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        predictions = model.generate(batch, max_new_tokens=max_new_tokens)
    
    predicted_emotion = predictions[0].strip().lower()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"File: {file_name}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Predicted: {predicted_emotion}")
    
    # Check if prediction is correct
    is_correct = predicted_emotion.startswith(ground_truth) or ground_truth.startswith(predicted_emotion)
    if ground_truth != "unknown":
        print(f"Correct: {'✅ Yes' if is_correct else '❌ No'}")
    print(f"{'='*60}\n")
    
    return {
        "file_path": eeg_file_path,
        "file_name": file_name,
        "ground_truth": ground_truth,
        "predicted": predicted_emotion,
        "is_correct": is_correct if ground_truth != "unknown" else None,
        "eeg_shape": eeg_data.shape
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single EEG file using trained OpenTSLMFlamingo model"
    )
    parser.add_argument(
        "--eeg_file",
        type=str,
        required=True,
        help="Path to the EEG CSV file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/gemma_3_270m/OpenTSLMFlamingo/stage6_brain_eeg/checkpoints/best_model.pt",
        help="Path to the model checkpoint (default: results/gemma_3_270m/OpenTSLMFlamingo/stage6_brain_eeg/checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps). If not specified, will auto-detect."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Load model
    model = load_model(args.checkpoint, device=device)
    
    # Run prediction
    result = predict_emotion(
        model=model,
        eeg_file_path=args.eeg_file,
        max_new_tokens=args.max_tokens,
        device=device
    )
    
    print("\n✅ Inference completed successfully!")


if __name__ == "__main__":
    main()
