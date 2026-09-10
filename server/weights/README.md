# Machine Learning Weights Directory

This directory stores the trained neural network weights and calibrated parametric models for the **Spatio-Temporal ConvLSTM2D U-Net** (`unet_burn.h5` & `unet_flood.h5`) and the **Multivariate Flood Susceptibility Regressor** (`flood_risk_model.json`).

## Model Architecture Specs
- **Model Name**: `ecopulse_spatiotemporal_unet`
- **Input Tensor**: `(Batch, Time=2, Height=256, Width=256, Channels=3)` (Pre-event vs Post-event Multi-Spectral Sentinel-2 / Landsat observation pairs)
- **Encoder**: 3-level TimeDistributed Convolutional Feature Pyramids (64 -> 128 -> 256 filters)
- **Temporal Bottleneck**: ConvLSTM2D recurrent convolutional gating with 512 filters
- **Decoder**: Transposed 2D Convolutions with skip connections from post-event observation frame
- **Output Tensor**: `(Batch, Height=256, Width=256, 1)` binary burn-scar / inundation probability mask
- **Hydrological Regressor**: Ridge-regularized multivariate model trained on `server/data/train.csv` (12 basin variables: rainfall, drainage, soil saturation, river management, deforestation, etc.)

## Training & Generating Weights
To generate or re-train weights locally:

```bash
python -m server.train
```

The model weights will be compiled and saved as `server/weights/unet_burn.h5` and `server/weights/flood_risk_model.json`.
