````markdown
# Workflow Optimization Guide

This guide explains the optimizations implemented to speed up the EEG emotion detection workflow.

## Overview

The workflow has been optimized in several key areas:
1. **Data Loading** - Parallel data loading with multiple workers
2. **Training** - Mixed precision training and gradient accumulation
3. **Validation** - Increased batch size for faster validation
4. **Inference** - Parallel file processing

## Configuration

All optimization settings are in `src/model_config.py`:

```python
# DataLoader optimizations
NUM_WORKERS = 4  # Number of worker processes for data loading
PIN_MEMORY = True  # Faster GPU transfer
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Mixed precision training
USE_MIXED_PRECISION = True  # Enable automatic mixed precision (AMP)
MIXED_PRECISION_DTYPE = "bf16"  # "bf16" (better) or "fp16" (wider support)

# Gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients over N steps

# Validation batch size
VAL_BATCH_SIZE = 4  # Increase from 1 for faster validation

# Model compilation
USE_TORCH_COMPILE = False  # Enable torch.compile (experimental)

# Inference optimizations
INFERENCE_BATCH_SIZE = 8
INFERENCE_NUM_WORKERS = 2
```

## Performance Improvements

### 1. Data Loading Optimization

**Before:**
- Single-threaded data loading (`num_workers=0`)
- No memory pinning
- Workers recreated each epoch

**After:**
- Multi-worker data loading (`num_workers=4`)
- Memory pinning for faster GPU transfer
- Persistent workers (reused between epochs)

**Expected Speedup:** 2-4x faster data loading

### 2. Mixed Precision Training

**Before:**
- Full FP32 precision (slower, uses more memory)

**After:**
- BF16/FP16 mixed precision (faster, uses less memory)
- Automatic loss scaling for FP16

**Expected Speedup:** 1.5-2x faster training, ~50% memory reduction

**Note:** BF16 is preferred on modern GPUs (A100, H100) as it's more stable than FP16.

### 3. Gradient Accumulation

**Before:**
- Batch size limited by GPU memory

**After:**
- Accumulate gradients over multiple steps
- Effective batch size = `batch_size × gradient_accumulation_steps`

**Benefits:**
- Train with larger effective batch sizes
- Better gradient estimates
- More stable training

**Example:** With `BATCH_SIZE=4` and `GRADIENT_ACCUMULATION_STEPS=4`, effective batch size = 16

### 4. Validation Optimization

**Before:**
- Validation batch size = 1 (very slow)

**After:**
- Validation batch size = 4 (configurable)

**Expected Speedup:** 3-4x faster validation

### 5. Model Compilation (Experimental)

**Before:**
- Standard PyTorch execution

**After:**
- `torch.compile()` for optimized execution (PyTorch 2.0+)

**Expected Speedup:** 10-30% faster inference (varies by model)

**Note:** Currently disabled by default. Enable by setting `USE_TORCH_COMPILE = True` in `model_config.py`.

### 6. Inference Optimization

**Before:**
- Sequential file processing

**After:**
- Parallel file I/O processing
- Configurable number of workers

**Expected Speedup:** 2-3x faster batch inference (for I/O-bound operations)

## Tuning Recommendations

### For Small GPUs (8GB VRAM)
```python
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
USE_MIXED_PRECISION = True
MIXED_PRECISION_DTYPE = "fp16"  # FP16 uses less memory than BF16
NUM_WORKERS = 2
VAL_BATCH_SIZE = 2
```

### For Medium GPUs (16GB VRAM)
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8
USE_MIXED_PRECISION = True
MIXED_PRECISION_DTYPE = "bf16"
NUM_WORKERS = 4
VAL_BATCH_SIZE = 4
```

### For Large GPUs (24GB+ VRAM)
```python
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = 8
USE_MIXED_PRECISION = True
MIXED_PRECISION_DTYPE = "bf16"
NUM_WORKERS = 8
VAL_BATCH_SIZE = 8
USE_TORCH_COMPILE = True  # Try enabling compilation
```

## Monitoring Performance

### Training Speed
Monitor the training loop progress bar:
- **Before optimization:** ~X seconds per batch
- **After optimization:** ~Y seconds per batch (should be faster)

### Memory Usage
Check GPU memory usage:
```python
import torch
print(torch.cuda.memory_summary())
```

### Data Loading Speed
Watch for "DataLoader bottleneck" warnings. If data loading is slow:
- Increase `NUM_WORKERS` (but not more than CPU cores)
- Enable `PERSISTENT_WORKERS = True`
- Check disk I/O speed (SSD recommended)

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `BATCH_SIZE`
2. Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
3. Enable mixed precision (`USE_MIXED_PRECISION = True`)
4. Reduce `NUM_WORKERS` (each worker uses memory)

### Slow Training
1. Check if data loading is the bottleneck (watch for gaps in GPU utilization)
2. Increase `NUM_WORKERS` if CPU-bound
3. Enable mixed precision
4. Try `USE_TORCH_COMPILE = True` (experimental)

### Validation Taking Too Long
1. Increase `VAL_BATCH_SIZE` (but ensure it fits in memory)
2. Reduce validation frequency (validate every N epochs)

### Inference Issues
1. If parallel processing causes errors, disable with `--no-parallel`
2. Reduce `--max_workers` if system is overloaded

## Expected Overall Speedup

With all optimizations enabled:
- **Training:** 2-3x faster
- **Validation:** 3-4x faster
- **Inference:** 2-3x faster (for batch processing)

**Total workflow speedup:** Approximately **2-3x faster** end-to-end.

## Best Practices

1. **Start conservative:** Begin with default settings and tune based on your hardware
2. **Monitor GPU utilization:** Use `nvidia-smi` or `torch.cuda.memory_summary()`
3. **Profile bottlenecks:** Use PyTorch profiler to identify slow operations
4. **Test incrementally:** Enable optimizations one at a time to measure impact
5. **Balance speed vs. stability:** Some optimizations (like torch.compile) may cause issues

## Advanced Optimizations

### For Multi-GPU Training
The code already supports distributed training. Use:
```bash
torchrun --nproc_per_node=4 curriculum_learning.py --model OpenTSLMFlamingo
```

### For Production Deployment
1. Enable `USE_TORCH_COMPILE = True` for inference
2. Use TensorRT or ONNX Runtime for further acceleration
3. Implement model quantization (INT8) for faster inference

## References

- [PyTorch DataLoader Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [torch.compile Documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)


````
