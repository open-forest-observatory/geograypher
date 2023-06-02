# semantic-mesh-pytorch3d

## Installation

### Install pytorch3d

Begin by following the instructions to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```
conda create -n semantic-meshes python=3.9
conda activate semantic-meshes
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
```

```
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
```

```
conda install pytorch3d -c pytorch3d
```

Validate the installation

```
python -c "import torch; print(torch.cuda.is_available())"
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

### Install pyvista

```
conda install -c conda-forge pyvista
```

```
python -c "import pyvista; print(pyvista.__version__)"
```

### install the project

```
poetry install
```
