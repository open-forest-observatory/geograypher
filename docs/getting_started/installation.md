# Installation

There are two ways to use this tool. If you are an internal collaborator working on the `JetStream2` cloud compute environment with access to the `/ofo-share` , you can directly use an existing `conda` environment. Note that this option is only suitable if you want to use the existing functionality and not make changes to the toolkit code or dependencies. If you are an external collaborator/user or want to do development work, please create your own new environment.

## Using existing environment

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

## Creating a new environment

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

## Install Geograypher:
```
pip install geograypher
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
