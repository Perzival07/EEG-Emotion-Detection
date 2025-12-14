# Brain EEG Emotion Classification Dataset

This dataset integrates the BrainDataset folder structure into OpenTSLM for emotion classification from EEG signals.

## Dataset Structure

The BrainDataset folder should contain emotion-labeled folders with CSV files:
```
BrainDataset/
├── Angry/
│   ├── angry1.csv
│   ├── angry2.csv
│   └── ...
├── Baseline/
│   ├── baseline1.csv
│   └── ...
├── Frightened/
├── Happy/
├── Protected/
├── Sad/
├── Satisfied/
├── Surprise/
└── Unconcerned/
```

## CSV File Format

Each CSV file contains:
- Row 1: Metadata header
- Row 2: Column headers
- Row 3+: EEG data with columns:
  - `EEG.AF3`, `EEG.F7`, `EEG.F3`, `EEG.FC5`, `EEG.T7`, `EEG.P7`, `EEG.O1`
  - `EEG.O2`, `EEG.P8`, `EEG.T8`, `EEG.FC6`, `EEG.F4`, `EEG.F8`, `EEG.AF4`

## Usage

### Training with Brain EEG Dataset

To train on the Brain EEG dataset as a standalone stage:

```bash
cd OpenTSLM-main
python curriculum_learning.py --model OpenTSLMSP --stages stage6_brain_eeg
```

Or include it in the full curriculum:

```bash
python curriculum_learning.py --model OpenTSLMSP --stages stage1_mcq stage2_captioning stage6_brain_eeg
```

### Testing the Dataset Loader

You can test the dataset loader directly:

```python
from time_series_datasets.brain_eeg.BrainEEGQADataset import BrainEEGQADataset

# Load the dataset
train_dataset = BrainEEGQADataset(split="train", EOS_TOKEN="<|endoftext|>")
val_dataset = BrainEEGQADataset(split="validation", EOS_TOKEN="<|endoftext|>")
test_dataset = BrainEEGQADataset(split="test", EOS_TOKEN="<|endoftext|>")

print(f"Train: {len(train_dataset)} samples")
print(f"Validation: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
```

## Dataset Details

- **Emotion Labels**: angry, baseline, frightened, happy, protected, sad, satisfied, surprise, unconcerned
- **EEG Channels**: 14 channels from EPOCX headset
- **Data Split**: 80% train, 10% validation, 10% test (stratified by emotion)
- **Task**: Multi-class emotion classification from EEG signals

## Model Configuration

The Brain EEG stage uses:
- **Epochs**: 30
- **Learning Rates**: 
  - Encoder: 2e-4 (OpenTSLMSP)
  - Projector: 1e-4 (OpenTSLMSP)
  - Base: 2e-4 (OpenTSLMFlamingo)
- **Metric**: Accuracy
- **LoRA**: Enabled for OpenTSLMSP (stages 3+)

## Notes

- The dataset automatically finds the BrainDataset folder relative to OpenTSLM-main
- Each EEG channel is treated as a separate time series
- Data is normalized per channel (mean=0, std=1)
- Missing channels are filled with zeros

