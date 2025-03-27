# residue-estimator

## Overview

The aim of this project is to develop a machine learning model to estimate soil crop residue levels from images. Designed for integration into a mobile application, this model will provide a quick, accessible alternative to traditional residue estimation methods, like the line-transect method.

The following segmentation methods are to be trained and tested for residue segmentation to see which yields the best results.
* U-Net
* SegFormer
* YOLO

## Contributing

We rely heavily on the [Kamiak High Performance Computing Cluster (HPC)](https://hpc.wsu.edu/kamiak-hpc/what-is-kamiak/) for our workflow. As this is a WSU research project, this guide assumes you have access to the Kamiak HPC and will utilize it as well.

### Prerequisites

Some useful resources in setting up your workflow:
* [Remote Development using SSH in VSCode](https://code.visualstudio.com/docs/remote/ssh)
* [Kamiak Quick Start Guide](https://hpc.wsu.edu/users-guide/quick-start-guide/)
* [Kamiak Cheat Sheet](https://wpcdn.web.wsu.edu/wp-research/uploads/sites/2940/2024/02/kamiak_cheat_sheet_vs1.pdf)

#### Requirements
* [Access to Kamiak HPC](https://hpc.wsu.edu/kamiak-hpc/requesting-access/)
* [VSCode](https://code.visualstudio.com/)
* [Remote - SSH (VSCode Extension)](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)

#### Setup
1. Install and set up the [requirements](#requirements)
2. In VSCode, set up and connect to the Kamiak remote host
3. Clone the repository:
```bash
git clone https://github.com/brandonxu360/residue-estimator.git
cd residue-estimator
```
4. Create and activate the conda environment:
```bash
# Load anaconda module (required on Kamiak)
module load anaconda3

# Create the conda environment using the provided environment.yml file
conda env create -f environment.yml
```
5. Create the data folder in the root directory of the project. Download `images_512` into the `data` folder. The data can be found [here](https://emailwsu-my.sharepoint.com/:f:/r/personal/a_norouzikandelati_wsu_edu/Documents/Ph.D/Projects/Residue_estimator_app/Dataset/images_512?csf=1&web=1&e=yU5RHG) (you may need to request access)

### Usage

#### Training
To submit a training job using SLURM:
```bash
sbatch jobs/[train_script_name].sbatch [ENV_NAME]
```
Where `ENV_NAME` is the optional name of the conda environment (defaults to `ml-hpc-env`)

#### Testing
To submit a testing job using SLURM:
```bash
sbatch jobs/[test_script_name].sbatch [ENV_NAME]
```
Where:
* `ENV_NAME` is the optional name of the conda environment (defaults to `ml-hpc-env`)
* `MODEL_JOB_ID` is the job ID for the model to test

#### Outputs
Model checkpoints and logs are saved to the `results/` and `logs/` directories, respectively.


