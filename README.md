# residue-estimator

This repository contains the code for developing a machine learning model to estimate soil crop residue levels from images. Designed for integration into a mobile application, this model aims to provide a quick, accessible alternative to traditional residue estimation methods, like the line-transect method.

## Installation

This project is containerized with Docker; to set up and run this project, it is recommended to have [Docker](https://www.docker.com/products/docker-desktop) installed on your machine. This will standardize the development environment and make the use of HPC resources easier.

> **Note**: The commands in this guide are provided in Bash and were run from the repository directory on an Ubuntu system.

### 1. Clone the Repository

```bash
git clone https://github.com/brandonxu360/residue-estimator.git
cd residue-estimator
```

### 2. Set Up Data Directory

- Ensure you have the necessary image data stored in a local folder. By default, this is expected to be in the `data/` directory at the root level of the project.
- The image data can be found at [here](https://emailwsu-my.sharepoint.com/:f:/r/personal/kirtir_wsu_edu/Documents/Projects_Agroecosystems/Brandon/Images/working_images/images?csf=1&web=1&e=P4VUv1). You may need to request access from maintainers.
- Inside the `data/` folder, the structure should precisely follow that of the online data source. It is easiest to simply download the data into the /data folder. In general, you can expect:
  - `data/images/label` — containing the labeled images
  - `data/images/original` — containing the original images

> **Note:** If your data is stored elsewhere, you will need to provide the correct location when running the container.

### 3. Pull/Build the Docker Image
Our Docker images can be found on our [Docker Hub Repository](https://hub.docker.com/r/brandonxu/residue-estimator-model).

**Recommended**: Instead of building the Docker image from scratch, you can pull it directly from Docker Hub using the following command:

```bash
docker pull brandonxu/residue-estimator-model:version1.1
```
This is recommended to guarantee image consistency.

Alternatively, use the following command to build the Docker image from the Dockerfile:

```bash
docker build -t brandonxu/residue-estimator-model:version1.1 .
```

### 4. Run the Container

After pulling the Docker image, run the container with the following command, mapping your local data directory to the container:

```bash
docker run --rm -v $(pwd)/data:/app/data brandonxu/residue-estimator-model:version1.1
```

For interactive mode, use:
```bash
sudo docker run --rm -u $(id -u):$(id -g) -it -v "$(pwd)/data:/app/data" brandonxu/residue-estimator-model:version1.1
```

> **Note**: If your data is stored in a different location, please replace `$(pwd)/data` with that location.

For more information about running containers, please visit the Docker docs: 
- [Docker Manual - Running Containers](https://docs.docker.com/engine/containers/run/)
- [Docker Docs Reference - Run](https://docs.docker.com/reference/cli/docker/container/run/)

## Maintainers

For access requests or further information, please contact:

- [Brandon Xu](mailto:brandon.xu@wsu.edu)
- [Amin Norouzi](mailto:a.norouzikandelati@wsu.edu)
