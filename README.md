# Geograypher: Multiview Semantic Reasoning with Geospatial Data

This tool is designed for multiview image datasets where multiple photos are taken of the same scene. The goal is to address two related tasks: generating a prediction about one point in the real world using observations of that point from multiple viewpoints and locating where a point in the real world is observed in each image. The intended application is drone surveys for ecology but the tool is designed to be generalizable.

In drone surveys, multiple overlapping images are taken of a region. A common technique to align these images is using photogrammetry software such as the commercially-available Agisoft Metashape or open-source COLMAP. This project only supports Metashape at the moment, but we plan to expand to other software. We use two outputs from photogrammetry, the location and calibration parameters of the cameras and a 3D "mesh" model of the environment. Using techniques from graphics, we can find the correspondences between locations on the mesh and on the image.

One task that this can support is multiview classification. For example, if you have a computer vision model that generates land cover classifications (for example trees, shrubs, grasses, and bare earth) for each pixel in an image, these predictions can be transferred to the mesh. Then, the predictions for each viewpoint can be aggregated using a voting or averaging scheme to come up with a final land cover prediction for that location. The other task is effectively the reverse. If you have the data from the field, for example marking one geospatial region as shrubs and another as grasses, you can determine which portions of each image corresponds to these classes. This information can be used to train a computer vision model, that could be used in the first step.


## Installation

There are two ways to use this tool. If you are an internal collaborator working on the `JetStream2` cloud compute environment with access to the `/ofo-share` , you can directly use an existing `conda` environment. Note that this option is only suitable if you want to use the existing functionality and not make changes to the toolkit code or dependencies. If you are an external collaborator/user or want to do development work, please create your own new environment.

### Using existing environment

This is for internal collaborators working on Jetstream2. Note that you should not make any changes to this environment since these changes will impact others. Only edits to my copy of the repository will be reflected when you import the tool. To begin, you must have installed `conda` on your Jetstream. Note that the following steps assums a conda config file already exists on your VM, and that it points to your local user's home directory for conda envs and pkgs. You can check this with `conda config --show` and look at the values under `pkgs_dirs` and `envs_dirs`. Then you can tell `conda` to look, secondarily, in the following additional places for environments and packages.

```
conda config --append envs_dirs /ofo-share/repos-david/conda/envs/
conda config --append pkgs_dirs /ofo-share/repos-david/conda/pkgs/
```

Now you should see all of my conda environments when you do `conda env list` . The one you want is `geograypher-stable` , and can be activated as follows:

```
conda activate geograypher-stable
```

Use this instead of `geograypher` in future steps.

### Creating a new environment

> For internal collaborators working on `/ofo-share`, you can opt to store the new environment on `/ofo-share` so that you can access it from any VM. First make sure the location where you want to store the env is set to the default for conda:
>
> ```
> conda config --prepend envs_dirs /ofo-share/repos-<yourname>/conda/envs/
> conda config --prepend pkgs_dirs /ofo-share/repos-<yourname>/conda/pkgs/
> ```
>
> You will need to run the above two lines on all new VMs before you can activate the env.

Create and activate the environment:

```
conda create -n geograypher python=3.9 -y
conda activate geograypher
```

> For internal collaborators working on /ofo-share, you could run into permissions issues when installing dependencies. Check that your executable permissions are valid by running python and python3.9.
>
> ```
> python
> python3.9
> ```
>
> If you get a permission denied error, get the location of the python executable inside of your conda environment (from the error message).
>
> Use the output and change the permissions using chmod. What the command should look like:
>
> ```
> chmod ugo+x <CONDA ENV LOCATION>/bin/python3.9
> ```

If you haven't already, install [poetry](https://python-poetry.org/docs/):

```
curl -sSL https://install.python-poetry.org | python3 -
```

Now use this to install the majority of dependencies. First `cd` to the directory containing the `geograypher` repo. Then run:

```
poetry install
```

Now install the `pytorch3d` dependencies that can't be installed with `poetry`:

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

If you are working on a headless machine, such as a remote server, you will need the [XVFB](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) package to provide a virtual frame buffer. This can be installed at the system level using the package manager, for example:
```
sudo apt install xvfb
```
If you do not have root access on your machine, it may not be possible to install xvfb.

You may get the following error when running `pyvista` visualization:

```
libGL error: MESA-LOADER: failed to open swrast: <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
```

If this happens, you can fix it by symlinking to the system version. I don't know why this is required.

```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 <CONDA ENV LOCATION>/lib/libstdc++.so.6
```

If you want to be able to call geograypher fucntions from Python (e.g. Jupyter notebooks), then you also need to install the Python module. You can install your local clone of the repo as a package:
```
pip install -e <path to your clone of the geograypher repo>
```

### Example data

The public example data is in `data/example_Emerald_Point_data` . You can run notebooks in the `examples` folder to see how to interact with this data. You can download this data using Google Drive from this [folder](https://drive.google.com/drive/folders/1gs5MkutQJEfg7tVnv01gzrf9NisAO5AT?usp=drive_link). Once you've downloaded it, extract it into the `data` folder.

### Using your own data

If you have a Metashape scene with the location of cameras, a mesh, and geospatial information, you can likely use geograypher. If you are using the Metashape GUI, you must do an important step before exporting the mesh model. Metashape stores the mesh in an arbitrary coordinate system that's optimized for viewing and will export it as such. To fix this, in the Metashape GUI you need to do `Model->Transform Object->Reset Transform` , then save the mesh with the local coordinates option. The cameras can be exported without any special considerations.

You can also use our scripted workflow for running Metashape, [automate-metashape](https://github.com/open-forest-observatory/automate-metashape). The cameras and the `local` mesh export will be properly formatted for use with geograypher.

### Running

There are currently two main 3D workflows that this tool supports, rendering and aggregation. The goal of rendering is to take data that is associated with a mesh or geospatially referenced and translate it to the viewpoint of each image. An example of this is exporting the height above ground or species classification for each point on an image. The goal of aggregation is to take information from each viewpoint and aggregate it onto a mesh and optionally export it as a geospatial file. An example of this is taking species or veg-cover type predictions from each viewpoints and aggregating them onto the mesh.

It also provides functionality for making predictions on top-down orthomosaics. This is not the main focus of the tool but is intended as a strong baseline or for applications where only this data is available.

There is one script for each of these workflows. They each have a variety of command line options that can be used to control the behavior. But in either case, they can be run without any flags to produce an example result. To see the options, run either script with the `-h` flag as seen below.

```
conda activate geograypher
python geograypher/entrypoints/render_labels.py --help
python geograypher/entrypoints/aggregate_viewpoints.py --help
python geograypher/entrypoints/orthomosaic_predictions.py --help
```

Quality metrics can be computed using the evaluation script

```
conda activate geograypher
python geograypher/entrypoints/evaluate_predictions.py --help
```
