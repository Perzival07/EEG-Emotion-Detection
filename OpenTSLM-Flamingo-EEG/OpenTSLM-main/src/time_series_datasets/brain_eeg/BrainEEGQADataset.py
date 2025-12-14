#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from datasets import Dataset
from typing import List, Tuple, Literal
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.brain_eeg.brain_eeg_loader import (
    load_brain_eeg_splits,
    EEG_CHANNELS,
    EMOTION_LABELS,
)

# Channel descriptions for prompts
CHANNEL_DESCRIPTIONS = [
    "The following is the EEG signal from the AF3 electrode (left frontal area)",
    "The following is the EEG signal from the F7 electrode (left frontal-temporal area)",
    "The following is the EEG signal from the F3 electrode (left frontal area)",
    "The following is the EEG signal from the FC5 electrode (left fronto-central area)",
    "The following is the EEG signal from the T7 electrode (left temporal area)",
    "The following is the EEG signal from the P7 electrode (left parietal area)",
    "The following is the EEG signal from the O1 electrode (left occipital area)",
    "The following is the EEG signal from the O2 electrode (right occipital area)",
    "The following is the EEG signal from the P8 electrode (right parietal area)",
    "The following is the EEG signal from the T8 electrode (right temporal area)",
    "The following is the EEG signal from the FC6 electrode (right fronto-central area)",
    "The following is the EEG signal from the F4 electrode (right frontal area)",
    "The following is the EEG signal from the F8 electrode (right frontal-temporal area)",
    "The following is the EEG signal from the AF4 electrode (right frontal area)",
]


class BrainEEGQADataset(QADataset):
    """
    Dataset for EEG emotion classification using OpenTSLM.
    Loads EEG data from BrainDataset folder structure.
    """
    
    def __init__(
        self, 
        split: Literal["train", "test", "validation"], 
        EOS_TOKEN: str, 
        format_sample_str: bool = False, 
        time_series_format_function=None,
        data_dir: str = None
    ):
        self.data_dir = data_dir
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)
    
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the Brain EEG dataset splits.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        return load_brain_eeg_splits(data_dir=self.data_dir)

    def _get_answer(self, row) -> str:
        """
        Get the answer (emotion label) from the row.
        
        Args:
            row: Dataset row
            
        Returns:
            Emotion label as a string
        """
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        """
        Get the pre-prompt text for emotion classification.
        
        Args:
            _row: Dataset row (unused)
            
        Returns:
            Pre-prompt text
        """
        text = """
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
"""
        return text.strip()

    def _get_post_prompt(self, _row) -> str:
        """
        Get the post-prompt text.
        
        Args:
            _row: Dataset row (unused)
            
        Returns:
            Post-prompt text
        """
        return "Emotion classification:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the EEG data into a list of TextTimeSeriesPrompt objects.
        Each EEG channel becomes a separate time series prompt.
        
        Args:
            row: Dataset row containing 'eeg_data' (numpy array of shape (n_channels, n_timepoints))
            
        Returns:
            List of TextTimeSeriesPrompt objects, one for each EEG channel
        """
        # Extract EEG data
        eeg_data = row["eeg_data"]
        
        # Convert to tensor (handle both numpy arrays and lists)
        if isinstance(eeg_data, np.ndarray):
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        elif isinstance(eeg_data, list):
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        else:
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        # eeg_tensor shape: (n_channels, n_timepoints)
        n_channels, n_timepoints = eeg_tensor.shape
        
        # Normalize each channel independently
        means = eeg_tensor.mean(dim=1, keepdim=True)  # (n_channels, 1)
        stds = eeg_tensor.std(dim=1, keepdim=True)    # (n_channels, 1)
        
        # Handle zero or very small standard deviations
        min_std = 1e-6
        stds = torch.clamp(stds, min=min_std)
        
        eeg_normalized = (eeg_tensor - means) / stds
        
        # Check for NaN/Inf
        if torch.isnan(eeg_normalized).any() or torch.isinf(eeg_normalized).any():
            print(f"Warning: NaN/Inf detected in EEG data, file: {row.get('file_path', 'unknown')}")
            eeg_normalized = torch.nan_to_num(eeg_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create TextTimeSeriesPrompt for each channel
        prompts = []
        for i, (channel_name, description) in enumerate(zip(EEG_CHANNELS, CHANNEL_DESCRIPTIONS)):
            if i >= n_channels:
                break
            
            channel_data = eeg_normalized[i].tolist()
            mean_val = means[i].item()
            std_val = stds[i].item()
            
            text_prompt = f"{description}, it has mean {mean_val:.4f} and std {std_val:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, channel_data))
        
        return prompts

    @staticmethod
    def get_labels() -> List[str]:
        """
        Return the list of all possible emotion labels.
        """
        return [label.lower() for label in EMOTION_LABELS]

    def _format_sample(self, row):
        """Override to add additional metadata."""
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["file_path"] = row.get("file_path", "")
        return sample


if __name__ == "__main__":
    # Test the dataset
    print("Testing BrainEEGQADataset...")
    
    train_dataset = BrainEEGQADataset(split="train", EOS_TOKEN="<|endoftext|>")
    val_dataset = BrainEEGQADataset(split="validation", EOS_TOKEN="<|endoftext|>")
    test_dataset = BrainEEGQADataset(split="test", EOS_TOKEN="<|endoftext|>")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Show sample data
    if len(train_dataset) > 0:
        print("\n" + "="*50)
        print("Sample data from training set:")
        sample = train_dataset[0]
        print(f"  Keys: {sample.keys()}")
        print(f"  Label: {sample['label']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Pre-prompt length: {len(sample['pre_prompt'])}")
        print(f"  Post-prompt: {sample['post_prompt']}")
        print(f"  Number of time series: {len(sample['time_series_text'])}")
        if len(sample['time_series_text']) > 0:
            print(f"  First time series text length: {len(sample['time_series_text'][0])}")

