# WANDR: Intention-guided Human Motion Generation

Welcome to the WANDR (CVPR-24) project repository! Thank you for your interest in our work. Fell free to open issues or email me about any suggestions/problems you encounter.
This guide will help you set up and get started with the project.

## Introduction

This initial commit allows you to download and run our model. Soon code and data for training and evaluating will follow. Stay tuned!

## Authors

- Markos Diomataris (1,2)
- Nikos Athanasiou (1)
- Omid Taheri (1)
- Xi Wang (2)
- Otmar Hilliges (2)
- Michael J. Black (1)

1. Max Planck Institute for Intelligent Systems, Tübingen, Germany
2. ETH Zürich, Switzerland

## Paper and Video

- [Read our paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Diomataris_WANDR_Intention-guided_Human_Motion_Generation_CVPR_2024_paper.pdf)
- [Watch our video](https://youtu.be/9szizM-XUCg)

## Getting Started

Follow these steps to set up your environment and download the necessary resources.

### 1. Download the SMPL-X Body Model

1. Visit the [SMPL-X website](https://smpl-x.is.tue.mpg.de/index.html).
2. Create an account if you don't have one.
3. Navigate to the "Download" section.
4. Download `SMPL-X v1.1` and place it in the root directory of this project.

### 2. Run the Setup Script

Run the `scripts/setup.sh` script to set up your environment:

```bash
sh scripts/setup.sh
```

### This script will:

1. Extract the previously downloaded SMPL-X zip file and place the contents under `./data/body_models`.
2. Set up a Python virtual environment named `wandr_env`.

### Activate the Virtual Environment

After running the setup script, activate the virtual environment with:

```bash
source wandr_env/bin/activate
```

### Run the demo

```bash
python demo.py
```

This should generate an ```output.mp4``` rendering of the produced motion.

## Citation

```bibtex
@inproceedings{diomataris2024wandr,
  title = {{WANDR}: Intention-guided Human Motion Generation},
  author = {Diomataris, Markos and Athanasiou, Nikos and Taheri, Omid and Wang, Xi and Hilliges, Otmar and Black, Michael J.},
  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
}
```


### Contact

This repository was implemented by [Markos Diomataris](https://ps.is.mpg.de/person/mdiomataris). Contact me at [markos.diomataris@tuebingen.mpg.de](mailto:markos.diomataris@tuebingen.mpg.de)