# semantic-mesh-pytorch3d

## Installation

### Install pytorch3d
Pytorch3D is a bit challenging to install, so we do it manually first. Begin by following the instructions to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```
conda create -n semantic-meshes python=3.9 -y
conda activate semantic-meshes
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
```

Now install the library itself

```
conda install pytorch3d -c pytorch3d -y
```

Validate the installation

```
python -c "import torch; print(torch.cuda.is_available())"
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

### install the rest of the dependencies and the project.
If you haven't already, install [poetry](https://python-poetry.org/docs/).

```
poetry install
```
For some reason, poetry may not work if it's not in a graphical session the first time. I think some form of authentication token is managed differently.

You may get the following error when running `pyvista` visualization:
```
libGL error: MESA-LOADER: failed to open swrast: <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
```
If this happens, you can fix it by symlinking to the system version. I don't know why this is required.
```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 <CONDA ENV LOCATION>/bin/../lib/libstdc++.so.6
```

### Example data
Data for this project is managed by a tool called [DVC](https://dvc.org/doc/install). This serves as an interface between git-tracked pointer files and the raw data. In our case, the data is hosted on Google Drive and you must have access to this [folder](https://drive.google.com/drive/folders/1vnnodshTUeZ6zXPdSdr9OPpYTU1dVdzk). If you are a collaborator and need the data, contact <djrussell@ucdavis.edu>.

The website provides a good overview of how to use `dvc`. In most cases, all you need to do is `dvc pull <filename>` to obtain the data.

### Running
Currently the only entrypoint is `semantic_mesh_pytorch3d/entrypoints/mesh_render.py`. The command line interface provides several options that are described there.
