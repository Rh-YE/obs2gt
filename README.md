# obs2gt — Observation → Ground-Truth Image Reconstruction (Ground-based Simulation)

Starting from SKIRT galaxy cubes, we forward-simulate ground-based r-band
observations and train a variational autoencoder (VAE) to reconstruct the
**noiseless ground truth (GT)** from the **noisy observation (OBS)**.

This repository contains all the **simulation** and **training** code required
to reproduce the reconstruction results, together with the underlying model
library. (Result evaluation / plotting scripts are not part of this repository.)

## Layout

```
obs2gt/
├── main.py                       # Training entry point (PyTorch Lightning / OmegaConf)
├── configs/generation/           # Three experiment configs (same network, different loss)
│   ├── ground_mae_l.yaml         #   L1 / MAE supervision
│   ├── ground_mse_l.yaml         #   MSE supervision
│   └── ground_chi2_l_obsivar.yaml#   Observation-variance-weighted χ² supervision
├── ground_sim/                   # Forward simulation (dataset generation)
│   ├── forward_model.py          #   SKIRT → ground-based observation forward model (OBS/GT/SIGMA)
│   └── build_dataset.py          #   Batch builder for train/valid/test datasets
└── sgm/                          # Model library (AutoencodingEngine, Encoder/Decoder,
                                  # Regularizer, DiscVAELoss, DataModule, Dataset, ...)
```

## Pipeline

### 1. Generate the simulated dataset

`ground_sim/forward_model.py` implements the full forward chain: luminosity-distance
flux scaling, angular-size scaling, Moffat PSF convolution, sky background (with a
tilt residual), multi-exposure Poisson + read noise, and a background and variance
that are **estimated from the data itself** (GT is never used to build SIGMA).

```bash
python -m ground_sim.build_dataset --out-root <dataset-output-dir> --workers 250
```

Each observation yields 4 random 64×64 cutouts. Data layout:

```
<root>/{train,valid,test}/GROUND_R/
    BGSUB/<name>.fits     # HDU0 = OBS (nanomaggy, background-subtracted)
                          # test additionally has HDU1 "GT" + HDU2 "COVERAGE"
    INVVAR/<name>.fits    # HDU0 = 1/SIGMA²
```

> The train/valid BGSUB files **contain no GT**, so reconstruction error cannot
> leak into the training/validation loss by construction.

### 2. Training

`AutoencodingEngine` in `sgm/models/autoencoder.py` takes a single-channel 64×64
OBS image and outputs a reconstruction; the encoder uses `cat_invvar=False`
(SIGMA only enters the loss weighting, never the network input).
Large size: `ch=128, z_channels=80, ch_mult=[1,2,4]`.

```bash
python main.py --base configs/generation/ground_mae_l.yaml  --train --gpus 0,
python main.py --base configs/generation/ground_mse_l.yaml  --train --gpus 0,
python main.py --base configs/generation/ground_chi2_l_obsivar.yaml --train --gpus 0,
```

The three configs differ only in `loss_type` (`mae` / `mse` / `chi2`); the
network architecture and data are identical.

## Notes

- The three large-tier models share the same network and the same data and differ
  only in the loss, in order to compare the effect of MAE / MSE / observation-
  variance-weighted χ² on reconstruction quality (seed 42).
- The dataset paths in the training configs are absolute paths from the
  experiment machine; adjust `data.params.train.dataset.path` and
  `data.params.validation.dataset.path` as needed when moving elsewhere.

## Environment

See `requirements.txt` (Python 3.10+, CUDA build of PyTorch).
