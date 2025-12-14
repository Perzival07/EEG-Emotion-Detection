#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# EEG channel names (14 channels from EPOCX headset)
EEG_CHANNELS = [
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1",
    "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
]

# Emotion labels (folder names)
EMOTION_LABELS = [
    "Angry", "Baseline", "Frightened", "Happy", "Protected", 
    "Sad", "Satisfied", "Surprise", "Unconcerned"
]

TEST_FRAC = 0.1
VAL_FRAC = 0.1

# Target sequence length for center cropping (to fit in GPU memory)
# 2048 time points = 512 patches (with patch_size=4)
# Reduced from 4096 to fit Gemma-3-270m in 8GB VRAM
TARGET_SEQUENCE_LENGTH = 2048


def center_crop_eeg(eeg_array: np.ndarray, target_length: int = TARGET_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Extract a fixed-length window from the center of the EEG sequence.
    
    Args:
        eeg_array: numpy array of shape (n_channels, n_timepoints)
        target_length: desired length of the output sequence
        
    Returns:
        numpy array of shape (n_channels, target_length) with center-cropped data
    """
    n_channels, n_timepoints = eeg_array.shape
    
    # If sequence is shorter than target, pad with zeros
    if n_timepoints < target_length:
        pad_width = target_length - n_timepoints
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(eeg_array, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    # If sequence is longer, extract center portion
    elif n_timepoints > target_length:
        center = n_timepoints // 2
        start = center - (target_length // 2)
        end = start + target_length
        return eeg_array[:, start:end]
    
    # If exactly the target length, return as is
    else:
        return eeg_array


def load_eeg_from_csv(csv_path: str, apply_center_crop: bool = True) -> np.ndarray:
    """
    Load EEG data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        apply_center_crop: Whether to apply center cropping to fixed length
        
    Returns:
        numpy array of shape (n_channels, n_timepoints) with EEG data
    """
    try:
        # Read CSV, skipping the first row (metadata) and using second row as header
        df = pd.read_csv(csv_path, skiprows=1)
        
        # Extract EEG channel columns
        eeg_data = []
        for channel in EEG_CHANNELS:
            if channel in df.columns:
                eeg_data.append(df[channel].values)
            else:
                # If channel is missing, fill with zeros
                print(f"Warning: Channel {channel} not found in {csv_path}, filling with zeros")
                eeg_data.append(np.zeros(len(df)))
        
        # Stack into array: (n_channels, n_timepoints)
        eeg_array = np.array(eeg_data, dtype=np.float32)
        
        # Remove NaN and Inf values
        eeg_array = np.nan_to_num(eeg_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply center crop to standardize sequence length
        if apply_center_crop:
            eeg_array = center_crop_eeg(eeg_array)
        
        return eeg_array
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        raise


def load_brain_eeg_dataset(data_dir: str = None) -> List[Dict]:
    """
    Load all EEG data from BrainDataset folder structure.
    
    Args:
        data_dir: Path to BrainDataset folder. If None, looks for ../BrainDataset
        
    Returns:
        List of dictionaries with keys: 'eeg_data', 'label', 'file_path'
    """
    if data_dir is None:
        # Try to find BrainDataset relative to OpenTSLM-main
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: time_series_datasets/brain_eeg -> time_series_datasets -> src -> OpenTSLM-main
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        data_dir = os.path.join(base_dir, "..", "BrainDataset")
        data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"BrainDataset directory not found: {data_dir}")
    
    print(f"Loading EEG data from: {data_dir}")
    
    all_data = []
    
    # Iterate through each emotion folder
    for emotion_label in EMOTION_LABELS:
        emotion_dir = os.path.join(data_dir, emotion_label)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Emotion folder not found: {emotion_dir}")
            continue
        
        # Find all CSV files in the emotion folder
        csv_files = sorted(glob.glob(os.path.join(emotion_dir, "*.csv")))
        
        if len(csv_files) == 0:
            print(f"Warning: No CSV files found in {emotion_dir}")
            continue
        
        print(f"Loading {len(csv_files)} files from {emotion_label}...")
        
        for csv_file in tqdm(csv_files, desc=f"Loading {emotion_label}"):
            try:
                eeg_data = load_eeg_from_csv(csv_file)
                
                # Skip if data is empty or all zeros
                if eeg_data.size == 0 or np.all(eeg_data == 0):
                    print(f"Warning: Empty or zero data in {csv_file}, skipping")
                    continue
                
                all_data.append({
                    "eeg_data": eeg_data,
                    "label": emotion_label.lower(),  # Use lowercase for consistency
                    "file_path": csv_file,
                    "n_channels": eeg_data.shape[0],
                    "n_timepoints": eeg_data.shape[1]
                })
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
    
    print(f"Loaded {len(all_data)} EEG samples")
    return all_data


def load_brain_eeg_splits(
    data_dir: str = None, 
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load BrainDataset and split into train, validation, and test sets.
    Uses stratified splitting to ensure all emotion classes are represented.
    
    Args:
        data_dir: Path to BrainDataset folder
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train, validation, test) datasets
    """
    # Load all data
    all_data = load_brain_eeg_dataset(data_dir)
    
    if len(all_data) == 0:
        raise ValueError("No data loaded from BrainDataset")
    
    # Convert to DataFrame for easier splitting
    import pandas as pd
    df = pd.DataFrame(all_data)
    
    # Stratified split by label
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=seed,
        stratify=df['label']
    )
    
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_adj,
        random_state=seed + 1,
        stratify=train_val_df['label']
    )
    
    # Convert back to list of dicts and create Dataset objects
    # Note: numpy arrays need to be converted to lists for Dataset.from_list()
    def convert_numpy_to_list(data_list):
        converted = []
        for item in data_list:
            new_item = item.copy()
            if isinstance(new_item['eeg_data'], np.ndarray):
                new_item['eeg_data'] = new_item['eeg_data'].tolist()
            converted.append(new_item)
        return converted
    
    train_data = convert_numpy_to_list(train_df.to_dict('records'))
    val_data = convert_numpy_to_list(val_df.to_dict('records'))
    test_data = convert_numpy_to_list(test_df.to_dict('records'))
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """Get the distribution of labels in the dataset."""
    labels = dataset['label']
    import pandas as pd
    return dict(pd.Series(labels).value_counts())


if __name__ == "__main__":
    # Test the loader
    train, val, test = load_brain_eeg_splits()
    
    print("\nTrain label distribution:")
    print(get_label_distribution(train))
    
    print("\nValidation label distribution:")
    print(get_label_distribution(val))
    
    print("\nTest label distribution:")
    print(get_label_distribution(test))
    
    # Show a sample
    if len(train) > 0:
        sample = train[0]
        print(f"\nSample data:")
        print(f"  Label: {sample['label']}")
        print(f"  EEG shape: {sample['eeg_data'].shape}")
        print(f"  File: {sample['file_path']}")

