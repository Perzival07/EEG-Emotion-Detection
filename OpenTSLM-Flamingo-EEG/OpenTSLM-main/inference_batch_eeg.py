#!/usr/bin/env python3
"""
Batch inference script for Brain EEG emotion classification.
Processes multiple EEG files from a directory and calculates accuracy metrics.

Usage:
    python inference_batch_eeg.py --data_dir /path/to/BrainDataset
    python inference_batch_eeg.py --data_dir /path/to/BrainDataset --emotion Happy
    python inference_batch_eeg.py --data_dir /path/to/BrainDataset --max_samples 10
"""

import sys
import os
import argparse
import torch
import json
import glob
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from inference_single_eeg import load_model, predict_emotion
from time_series_datasets.brain_eeg.brain_eeg_loader import EMOTION_LABELS


def batch_predict(
    model,
    data_dir: str,
    emotion: str = None,
    max_samples: int = None,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> dict:
    """
    Run batch prediction on EEG files.
    
    Args:
        model: Trained model
        data_dir: Path to BrainDataset directory
        emotion: Specific emotion to process (None for all)
        max_samples: Maximum number of samples to process (None for all)
        max_new_tokens: Maximum tokens to generate
        device: Device to use
        
    Returns:
        Dictionary with results and metrics
    """
    # Determine which emotions to process
    if emotion:
        emotions_to_process = [emotion]
    else:
        emotions_to_process = EMOTION_LABELS
    
    all_results = []
    correct = 0
    total = 0
    
    # Process each emotion
    for emotion_label in emotions_to_process:
        emotion_dir = os.path.join(data_dir, emotion_label)
        
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Warning: Emotion folder not found: {emotion_dir}")
            continue
        
        # Get all CSV files
        csv_files = sorted(glob.glob(os.path.join(emotion_dir, "*.csv")))
        
        if len(csv_files) == 0:
            print(f"⚠️  Warning: No CSV files found in {emotion_dir}")
            continue
        
        # Limit samples if specified
        if max_samples:
            csv_files = csv_files[:max_samples]
        
        print(f"\n{'='*60}")
        print(f"Processing {emotion_label}: {len(csv_files)} files")
        print(f"{'='*60}")
        
        # Process each file
        for csv_file in tqdm(csv_files, desc=f"Processing {emotion_label}"):
            try:
                result = predict_emotion(
                    model=model,
                    eeg_file_path=csv_file,
                    max_new_tokens=max_new_tokens,
                    device=device
                )
                
                all_results.append(result)
                
                if result["is_correct"] is not None:
                    total += 1
                    if result["is_correct"]:
                        correct += 1
                        
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
                continue
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    # Create summary
    summary = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "emotions_processed": emotions_to_process,
        "results": all_results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference on EEG files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/kd/Local Disk/GitHub/EEG-Emotion-Detection/OpenTSLM-Flamingo-EEG/BrainDataset",
        help="Path to BrainDataset directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/gemma_3_270m/OpenTSLMFlamingo/stage6_brain_eeg/checkpoints/best_model.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        choices=EMOTION_LABELS + [None],
        help="Specific emotion to process (default: all emotions)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per emotion (default: all)"
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
    parser.add_argument(
        "--output",
        type=str,
        default="batch_inference_results.json",
        help="Output JSON file for results (default: batch_inference_results.json)"
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
    
    # Run batch prediction
    summary = batch_predict(
        model=model,
        data_dir=args.data_dir,
        emotion=args.emotion,
        max_samples=args.max_samples,
        max_new_tokens=args.max_tokens,
        device=device
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Correct predictions: {summary['correct_predictions']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")
    print(f"{'='*60}\n")
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
