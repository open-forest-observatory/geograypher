# Geograypher: Enabling Geospatial Predictions from Individual Drone Images
[![DOI](https://zenodo.org/badge/644076369.svg)](https://zenodo.org/doi/10.5281/zenodo.11193026)

This tool is designed for multiview image datasets where multiple photos are taken of the same scene. The goal is to address two related tasks: generating a prediction about one point in the real world using observations of that point from multiple viewpoints and locating where a point in the real world is observed in each image. The intended application is drone surveys for ecology but the tool is designed to be generalizable.

In drone surveys, multiple overlapping images are taken of a region. A common technique to align these images is using photogrammetry software such as the commercially-available Agisoft Metashape or open-source COLMAP. This project only supports Metashape at the moment, but we plan to expand to other software. We use two outputs from photogrammetry, the location and calibration parameters of the cameras and a 3D "mesh" model of the environment. Using techniques from graphics, we can find the correspondences between locations on the mesh and on the image.

One task that this can support is multiview classification. For example, if you have a computer vision model that generates land cover classifications (for example trees, shrubs, grasses, and bare earth) for each pixel in an image, these predictions can be transferred to the mesh. Then, the predictions for each viewpoint can be aggregated using a voting or averaging scheme to come up with a final land cover prediction for that location. The other task is effectively the reverse. If you have the data from the field, for example marking one geospatial region as shrubs and another as grasses, you can determine which portions of each image corresponds to these classes. This information can be used to train a computer vision model, that could be used in the first step.

### Basic Installation
Internal Collaborators please navigate [here](https://docs.openforestobservatory.org/internal-docs/jetstream/#geograypher)

Create and activate a conda environment:

```
conda create -n geograypher -c conda-forge python=3.9 -y
conda activate geograypher
```

Install Geograypher:
```
pip install geograypher
```

Note the instructions above are suitable if you only need to use the exisiting functionality of Geograypher and not make changes to the toolkit code or dependencies. If you want to do development work please navigate [here](https://open-forest-observatory.github.io/geograypher/getting_started/installation/) for more instructions.

### Getting started

There are two main places to look when getting started, the `example` notebooks and the command line scripts in `geograypher/entrypoints`. You can start by using the  [example data](https://ucdavis.box.com/v/geograypher-example-data) and download it to the `data` folder of this project. The notebooks have paths that point to this example data by default, so once the data is downloaded they can be run without any modifications. The command line tools each provide a help (`--help/-h`) option that describes the purpose of the tool and the command line options.

### Documentation
Documentation of this tool, including how to set it up and examples of how to use it, can be found [here](https://open-forest-observatory.github.io/geograypher/).
