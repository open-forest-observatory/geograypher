# Installation

If you only need to use the exisiting functionality of Geograypher and not make changes to the toolkit code or dependencies, follow the `Basic Installation` instructions. 

If you want to do development work, please see `Advanced/Developer Installation`.

Internal collaborators please navigate [here](https://docs.openforestobservatory.org/internal-docs/) for more instructions. 

## Basic Installation
Create and activate a conda environment:

```
conda create -n geograypher python=3.9 -y
conda activate geograypher
```

Install Geograypher:
```
pip install geograypher
```

## Advanced/Developer Installation
Create and activate a conda environment:

```
conda create -n geograypher python=3.9 -y
conda activate geograypher
```

Install [poetry](https://python-poetry.org/docs/):

```
curl -sSL https://install.python-poetry.org | python3 -
```

Now use this to install the majority of dependencies. First cd to the directory containing the `geograypher` repo. Then run:

```
poetry install
```

You may get the following error when running `pyvista` visualization:

```
libGL error: MESA-LOADER: failed to open swrast: <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
```

If this happens, you can fix it by symlinking to the system version. I don't know why this is required.

```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 <CONDA ENV LOCATION>/lib/libstdc++.so.6
```

## Working on Headless Machine
If you are working on a headless machine, such as a remote server, you will need the [XVFB](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) package to provide a virtual frame buffer. This can be installed at the system level using the package manager, for example:
```
sudo apt install xvfb
```
If you do not have root access on your machine, it may not be possible to install xvfb.

## Optional: Install `pytorch3d`
If you are working on a headless machine and are unable to install xvfb, `pytorch3d` can be a viable alternative since installing it does not require admin privileges.

Install the pytorch3d dependencies:

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
