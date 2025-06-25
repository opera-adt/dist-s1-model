# dist-s1-model

This is a repository that includes the transformer model and relevant training routines.
It is a greatly distilled version of Harris Hardiman-Mostow's research [repository](https://github.com/OPERA-Cal-Val/deep-dist-s1-research) with optimizations and improvements specifically tailored for the DIST-S1 product written by [Diego Martinez](https://github.com/dmartinez05). There are also additional notebooks to inspect the input dataset and visualize the model application to existing OPERA RTC data.

## Usage

### Training

1. Download the data at:
    - training data (~53 GB): `<url>`
    - test data (~13 GB): `<url>`
2. Update the train/test paths in the `trainer.py` script.
3. Install the environment via mamba: `mamba create env -f environment_gpu.yml`
4. Activate the environment and in this repository run `python trainer.py`. Or to preserve the standard io: `python trainer.py > trainer.out 2> trainer.err`.

### Application

See the notebooks. This is a work in progress!

## Data

We will create another repository for curating the SAR data. TODO.

## WandB

To post results to wandb, set the following flags in config.yaml to have the following values
use_wandb: True
wandb_entity: cmarshak-nasa-jet-propulsion-laboratory 
wandb_project: spatiotemporal-transformer

Note: before using wandb for the first time, you must open a terminal session, activate the env used to run dist-s1-model code, and run
wandb login 
The command line will prompt you for an API key that can be found at https://wandb.ai/home 

## References

- OPERA Disturbance Suite: https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/

- Hardiman-Mostow, Harris, Charles Marshak, and Alexander L. Handwerger. "Deep Self-Supervised Disturbance Mapping with the OPERA Sentinel-1 Radiometric Terrain Corrected SAR Backscatter Product." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2025).[arxiv](https://arxiv.org/abs/2501.09129)
