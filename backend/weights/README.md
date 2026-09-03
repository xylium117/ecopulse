# Machine Learning Weights Directory

This directory stores the pre-trained neural network weights for the **Spatio-Temporal ConvLSTM2D U-Net** (`unet_burn.h5`).

## Model Architecture Specs
- **Model Name**: `ecopulse_spatiotemporal_unet`
- **Input Tensor**: `(Batch, Time=2, Height=256, Width=256, Channels=3)` (Pre-event vs Post-event Multi-Spectral Sentinel-2 / Landsat observation pairs)
- **Encoder**: 3-level TimeDistributed Convolutional Feature Pyramids (64 -> 128 -> 256 filters)
- **Temporal Bottleneck**: ConvLSTM2D recurrent convolutional gating with 512 filters
- **Decoder**: Transposed 2D Convolutions with skip connections from post-event observation frame
- **Output Tensor**: `(Batch, Height=256, Width=256, 1)` binary burn-scar / deforestation probability mask

## Training & Generating Weights
To generate or re-train weights locally:

```bash
python -m backend.train
```

The model weights will be compiled and saved as `backend/weights/unet_burn.h5`.
