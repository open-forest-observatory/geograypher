# multiview-prediction-toolkit
This tools is designed for multi-view image datasets where multiple photos are taken of the same scene. The goal is to address two related tasks: generating a prediction about one point in the real world using observations of that point from multiple viewpoints and locating where a point in the real world is observed in each image. The intended application is drone surveys for ecology but the tools is designed to be generalizable.

In drone surveys, multiple overlapping images are taken of a region. A common technique to align these images is using photogrametry software such as the commercially-available Agisoft Metashape or open-source COLMAP. This project only supports Metashape at the moment, but we plan to expand to other software. We use two outputs from photogrametry, the location and calibration parameters of the cameras and a 3D "mesh" model of the environment. Using techniques from graphics, we can find the correspondences between locations on the mesh and on the image. 

One task that this can support is multi-view classification. For example, if you have a computer vision model that generates land cover classifications (for example trees, shrubs, grasses, and bare earth) for each pixel in an image, these predictions can be transfered to the mesh. Then, the predictions for each viewpoint can be aggregated using a voting or averging scheme to come up with a final land cover prediction for that location. The other task is task is effectively the reverse. If you have the data from the field, for example marking one geospatial region as shrubs and another as grasses, you can determine which portions of each image corresponds to these classes. This information can be used to train a computer vision model, that could be used in the first step.


## Installation

### Install pytorch3d
Pytorch3D is a bit challenging to install, so we do it manually first. Begin by following the instructions to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```
conda create -n multiview-prediction python=3.9 -y
conda activate multiview-prediction
```

If you haven't already, install [poetry](https://python-poetry.org/docs/). Now use this to install a large set of the dependencies.

```
poetry install
```

For some reason, poetry may not work if it's not in a graphical session. I think some form of authentication token is managed differently.

Now install the `pytorch3d` dependencies that can't be installed with `poetry`.

```
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
```

Validate the installation

```
python -c "import torch; print(torch.cuda.is_available())"
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

You may get the following error when running `pyvista` visualization:
```
libGL error: MESA-LOADER: failed to open swrast: <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
```
If this happens, you can fix it by symlinking to the system version. I don't know why this is required.
```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6
```

### Example data
Data for this project is managed by a tool called [DVC](https://dvc.org/doc/install). This serves as an interface between git-tracked pointer files and the raw data. In our case, the data is hosted on the CyVerse Data Store and you must have access to this [folder](https://de.cyverse.org/data/ds/iplant/home/shared/ofo/internal/DVC_test/multiview_prediction_toolkit_DVC). If you are a collaborator and need the data, contact <djrussell@ucdavis.edu>.

Then you need to add your account credentials with
```
dvc remote modify --local cyverse user <cyverse username>
dvc remote modify --local cyverse password <cyverse password>
```

Note that the local flag adds this information to a the `.dvc/config.local` file. Your password is now stored in plaintext on your machine, so use caution. This file is not tracked by git, or else you'd expose your passwords to the outside world.

Note that the CyVerse WebDav server is not particularly powerful so they requested we keep the number of workers to 5. This is the `jobs` field in the `.dvc/config` file.

The website provides a good overview of how to use `dvc`. In most cases, all you need to do is `dvc pull <filename>` to obtain the data.

### Running
There are currently two main 3D workflows that this tool supports, rendering and aggregation. The goal of rendering is to take data that is associated with a mesh or geospatially referenced and translate it to the viewpoint of each image. An example of this is exporting the height above ground or species classification for each point on an image. The goal of aggregation is to take information from each viewpoint and aggregate it onto a mesh and optionally export it as a geospatial file. An example of this is taking species or veg-cover type predictions from each viewpoints and aggregating them onto the mesh.

It also provides functionality for making predictions on top-down orthomosaics. This is not the main focus of the tool but is intended as a strong baseline or for applications where only this data is available.

There is one script for each of these workflows. They each have a variety of command line options that can be used to control the behavior. But in either case, they can be run without any flags to produce an example result. To see the options, run either script with the `-h` flag as seen below.
```
conda activate multiview-prediction
python multiview_prediction_toolkit/entrypoints/mesh_render.py --help
python multiview_prediction_toolkit/entrypoints/aggregate_viewpoints.py --help
python multiview_prediction_toolkit/entrypoints/orthomosaic_predictions.py --help
```

Quality metrics can be computed using the evaluation script
```
conda activate multiview-prediction
python multiview_prediction_toolkit/entrypoints/evaluate_predictions.py --help
```

There are also a variety of scripts in the `dev` folder. These are one-off examples designed for a specific task and are not designed to be robust, current, or generalizable. However, they can serve as an example for your own scripts.

