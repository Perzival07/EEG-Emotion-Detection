# EEG-Emotion-Detection

## 1. **Organise** your file in the following order


## Folder Structure

```bash
dataset/
│
├── Angry/
│   ├── angry1.csv
│   ├── angry2.csv
│   └── ...
├── Happy/
├── Sad/
├── Frightened/
├── Protected/
├── Satisfied/
├── Surprise/
└── Unconcerned/
```


## 2. **Load**, **Label**, and **Combine** All Files

Here’s a script to:
- Walk through each folder,
- Label each sample with its folder name (emotion),
- Merge everything into one DataFrame,
- Save as `emotion_dataset.csv`.

```python
import os
import pandas as pd

def load_all_emotion_data(root_folder):
    all_data = []

    for emotion in os.listdir(root_folder):
        emotion_folder = os.path.join(root_folder, emotion)
        if not os.path.isdir(emotion_folder):
            continue

        for file in os.listdir(emotion_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(emotion_folder, file)
                try:
                    df = pd.read_csv(file_path)
                    df['Emotion'] = emotion
                    all_data.append(df)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

    return pd.concat(all_data, ignore_index=True)

# Example usage:
root_data_folder = "/mnt/data/emotions_dataset"  # adjust to your actual path
final_df = load_all_emotion_data(root_data_folder)

# Optionally drop timestamp columns if not needed
final_df = final_df.drop(columns=['Timestamp', 'OriginalTimestamp'], errors='ignore')

# Save the final combined dataset
final_df.to_csv("emotion_dataset.csv", index=False)
```

