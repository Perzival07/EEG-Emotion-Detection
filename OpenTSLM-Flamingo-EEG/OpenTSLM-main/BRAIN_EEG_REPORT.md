# Brain EEG OpenTSLM System Report

## 1. Overview
This system is a specialized adaptation of **OpenTSLM** (Open Time Series Language Model) designed exclusively for **EEG-based Emotion Classification**. It processes raw EEG signals from 14 channels, integrates them with textual descriptions, and uses a Large Language Model (LLM) to classify the subject's emotional state.

## 2. System Architecture

The architecture consists of three main layers:

### A. Data Layer (`BrainEEGQADataset`)
*   **Input**: 14-channel EEG data (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4).
*   **Preprocessing**:
    *   **Normalization**: Each channel is independently normalized (zero mean, unit variance).
    *   **Prompt Engineering**: Converts EEG signals into a multimodal prompt.
    *   Structure:
        *   **Pre-prompt**: Context setting ("You are given EEG data... classify the emotional state...").
        *   **Time Series Prompts**: Sequential list of channel data with descriptions (e.g., "The following is the EEG signal from the AF3 electrode...").
        *   **Post-prompt**: Task instruction ("Emotion classification:").

### B. Model Layer (`OpenTSLM`)
*   **Backbone**: The system supports `OpenTSLMFlamingo` or `OpenTSLMSP` architectures.
*   **Encoder**: A transformer-CNN hybrid encoder processes the time-series patches.
*   **Projector**: Projects the encoded time-series features into the LLM's embedding space.
*   **LLM**: A Frozen Pre-trained LLM (e.g., Llama, Gemma) generates the classification (emotion label).
*   **Method**: The model uses **Curriculum Learning** to conceptually separate training stages, currently focused on `stage6_brain_eeg`.

### C. Training Pipeline (`curriculum_learning.py`)
*   **Orchestrator**: Manages the training loop, distributed training (DDP), and stage transitions.
*   **Optimization**:
    *   Uses `AdamW` optimizer.
    *   Supports **LoRA** (Low-Rank Adaptation) for efficient fine-tuning of the LLM.
    *   Tracks validation loss and saves the best model checkpoints.

## 3. Workflow Mechanism

1.  **Initialization**:
    *   The `CurriculumTrainer` initializes the model (`OpenTSLMFlamingo`/`SP`) and distributed backend.
    *   It prepares the `results/` directory structure for `stage6_brain_eeg`.

2.  **Data Loading**:
    *   `BrainEEGQADataset` loads train/val/test splits.
    *   Data flows into `DataLoader` which handles batching and padding (via `extend_time_series_to_match_patch_size_and_aggregate`).

3.  **Forward Pass**:
    *   **Tokenization**: Text is tokenized; Time series data is patched.
    *   **Encoding**: Time series patches are encoded into vector embeddings.
    *   **Fusion**: Text embeddings and Time series embeddings are interleaved/fused.
    *   **Generation**: The LLM predicts the next tokens, aiming to generate the correct emotion label (e.g., "happy", "sad").

4.  **Training Loop**:
    *   Iterates through epochs.
    *   Computes Loss (Cross-Entropy on generated tokens).
    *   Backpropagates gradients to update Encoder, Projector, and LoRA weights.
    *   Logs metrics to `loss_history.txt`.

5.  **Evaluation**:
    *   Runs on the Test set.
    *   Generates predictions.
    *   Compares against ground truth labels (accuracy).

## 4. File Structure (Post-Cleanup)

```text
OpenTSLM-main/
├── curriculum_learning.py       # Main entry point for training
├── requirements.txt             # Python dependencies
├── src/
│   ├── time_series_datasets/    # Dataset implementations
│   │   ├── brain_eeg/           # Brain EEG specific logic
│   │   │   ├── BrainEEGQADataset.py
│   │   │   └── brain_eeg_loader.py
│   │   ├── QADataset.py         # Base dataset class
│   │   └── util.py              # Data utilities (padding)
│   ├── model/                   # Model architecture definitions
│   └── open_flamingo/           # Submodule for Flamingo architecture
└── evaluation/                  # Evaluation scripts (cleaned)
```

## 5. How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Training**:
    ```bash
    python curriculum_learning.py --model_type OpenTSLMFlamingo --llm_id meta-llama/Llama-3.2-1B
    ```
    *Adjust `llm_id` and `model_type` as needed.*

3.  **Monitor**:
    *   Check `results/<llm_id>/<stage>/loss_history.txt` for progress.
