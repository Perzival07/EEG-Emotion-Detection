```markdown
# OpenTSLM Project Documentation: Function and Use of Files

This document provides a detailed overview of the files and directories in the `OpenTSLM-main` project, specifically focusing on the EEG Emotion Detection adaptation.

## Table of Contents
1. [Root Directory](#1-root-directory)
2. [Source Code (`src/`)](#2-source-code-src)
   - [Models (`src/model/`)](#src-model)
   - [Prompts (`src/prompt/`)](#src-prompt)
   - [Datasets (`src/time_series_datasets/`)](#src-time_series_datasets)
3. [Data and Results (`data/`, `results/`)](#3-data-and-results)
4. [Evaluation (`evaluation/`)](#4-evaluation)

---

## 1. Root Directory

- **`curriculum_learning.py`**: The main entry point for training the OpenTSLM models. It implements a `CurriculumTrainer` class that manages the multi-stage training process (TSQA MCQ, Captioning, EEG classification, etc.). It handles model initialization, data loading, training loops, evaluation, and checkpoint saving.
- **`inference_single_eeg.py`**: A dedicated script for running inference on a single EEG CSV file. It loads a trained model, preprocesses the EEG data, and generates an emotion prediction.
- **`inference_batch_eeg.py`**: A script for running batch inference on multiple EEG files in a directory. It reports overall accuracy and saves detailed results to a JSON file.
- **`add_headers_script.py`**: A utility script used to add standard project headers (license, credits) to source files.
- **`BRAIN_EEG_REPORT.md`**: A system report summarizing the architecture and workflow specifically for the Brain EEG emotion classification task.
- **`README.md`**: The main project documentation providing an overview of OpenTSLM, installation instructions, and quick-start commands.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.
- **`__init__.py`**: Makes the root directory a Python package.
- **`batch_inference_results.json`**: (Generated) Output from `inference_batch_eeg.py` containing prediction metrics.

---

## 2. Source Code (`src/`)

### Core Files
- **`src/data.py`**: Contains data loading utilities for the TSQA (Time Series Question Answering) dataset, including normalization and collation functions.
- **`src/logger.py`**: Implements a centralized, singleton-like `OpenTSLMLogger` for consistent logging across the project with support for different levels (info, debug, warning, success, etc.).
- **`src/model_config.py`**: Central configuration file for hyperparameters such as batch size, learning rates, patch size, and model dimensions.

### <a name="src-model"></a>Models (`src/model/`)

#### LLM Components (`src/model/llm/`)
- **`OpenTSLMFlamingo.py`**: Implementation of the OpenTSLM model using the Flamingo-style architecture. It uses cross-attention layers to fuse time-series tokens into the LLM.
- **`OpenTSLMSP.py`**: Implementation of the OpenTSLM model using a "Sequence Prefix" (SP) approach. It prepends time-series embeddings as tokens to the LLM's input.
- **`TimeSeriesLLM.py`**: An abstract base class defining the interface for time-series-enabled LLMs (e.g., `generate`, `compute_loss`).
- **`TimeSeriesFlamingoWithTrainableEncoder.py`**: A variant of the Flamingo architecture where the encoder is specifically marked as trainable.

#### Encoder Components (`src/model/encoder/`)
- **`TransformerCNNEncoder.py`**: A hybrid encoder that uses a 1D Convolutional layer for initial patching followed by a series of Transformer blocks.
- **`CNNTokenizer.py`**: A simplified version of the encoder that performs 1D convolution patching and adds positional embeddings, effectively "tokenizing" raw time-series.
- **`TransformerMLPEncoder.py`**: An encoder using a Linear layer for patching followed by Transformer blocks.
- **`TimeSeriesEncoderBase.py`**: Abstract base class for all time-series encoders.

#### Projector Components (`src/model/projector/`)
- **`LinearProjector.py`**: A single linear layer used to project encoder outputs into the LLM embedding space.
- **`MLPProjector.py`**: A Multi-Layer Perceptron (MLP) used for the same projection task, offering higher capacity.

### <a name="src-prompt"></a>Prompts (`src/prompt/`)

- **`prompt.py`**: Base class/interface for prompt components.
- **`text_prompt.py`**: Represents a simple text-only prompt.
- **`text_time_series_prompt.py`**: A specialized prompt chunk that links a specific piece of text (e.g., "AF3 electrode data:") with its corresponding raw time-series segment.
- **`full_prompt.py`**: A container class that aggregates a pre-prompt, multiple `TextTimeSeriesPrompt` objects, and a post-prompt into a single structure.
- **`prompt_with_answer.py`**: Extends `FullPrompt` to include the ground truth answer, used primarily for training and loss calculation.

### <a name="src-time_series_datasets"></a>Datasets (`src/time_series_datasets/`)

- **`QADataset.py`**: An abstract base class for Question-Answering datasets involving time-series. It handles data splitting and formatting into `PromptWithAnswer` objects.
- **`util.py`**: General utilities for data manipulation, such as padding or aggregation.
- **`constants.py`**: Stores dataset-wide constants.

#### Brain EEG Specifics (`src/time_series_datasets/brain_eeg/`)
- **`BrainEEGQADataset.py`**: Subclass of `QADataset` tailored for EEG data. It maps the 14 EEG channels into a multimodal prompt.
- **`brain_eeg_loader.py`**: Logic for loading raw EEG CSV files, performing center-cropping (to manage sequence length for VRAM), and generating stratified train/val/test splits.

---

## 3. Data and Results

- **`data/`**: Expected directory for raw datasets.
  - **`BrainDataset/`**: Contains subdirectories for each emotion class (Angry, Happy, etc.) with individual EEG CSV files.
- **`results/`**: Directory where training outputs are saved.
  - Organized as: `results/<llm_id>/<model_type>/<stage>/`
  - Contains `checkpoints/` (model weights) and `results/` (metrics and predictions).

---

## 4. Evaluation

- **`evaluation/`**: Contains utility scripts for evaluating models.
  - **`opentslm/`**: Specialized evaluation logic for OpenTSLM metrics.

```
