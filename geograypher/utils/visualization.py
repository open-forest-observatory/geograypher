import json
import logging
import typing
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from imageio import imread, imwrite
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from tqdm import tqdm

from geograypher.constants import NULL_TEXTURE_INT_VALUE, PATH_TYPE
from geograypher.utils.files import ensure_folder


def safe_start_xvfb():
    try:
        pv.start_xvfb()
    except OSError:
        logging.warning("Could not start xvfb because it's not supported on Windows")


def create_pv_plotter(
    off_screen: bool,
    force_xvfb: bool = False,
    plotter: typing.Union[None, pv.Plotter] = None,
):
    """Create a pyvista plotter while handling offscreen rendering

    Args:
        off_screen (bool):
            Whether the plotter should be offscreen
        force_xvfb (bool, optional):
            Should XVFB be used for rendering by default. Defaults to False.
        plotter ((None, pv.Plotter), optional):
            Existing plotter to use, will just return it if not None. Defaults to None
    """
    # If a valid plotter has not been passed in create one
    if not isinstance(plotter, pv.Plotter):
        # Catch the warning that there is not xserver running
        with warnings.catch_warnings(record=True) as w:
            # Create the plotter which may be onscreen or off
            plotter = pv.Plotter(off_screen=off_screen)

        # Start xvfb if requested or the system is not running an xserver
        if force_xvfb or (len(w) > 0 and "pyvista.start_xvfb()" in str(w[0].message)):
            # Start a headless renderer
            safe_start_xvfb()
    return plotter


def get_vis_options_from_IDs_to_labels(
    IDs_to_labels: typing.Union[None, dict],
    cmap_continous: str = "viridis",
    cmap_10_classes: str = "tab10",
    cmap_20_classes: str = "tab20",
    cmap_many_classes: str = "viridis",
):
    """Determine vis options based on a given IDs_to_labels object

    Args:
        IDs_to_labels (typing.Union[None, dict]): _description_
        cmap_continous (str, optional):
            Colormap to use if the values are continous. Defaults to "viridis".
        cmap_10_classes (str, optional):
            Colormap to use if the values are discrete and there are 10 or fewer classes. Defaults to "tab10".
        cmap_20_classes (str, optional):
            Colormap to use if the values are discrete and there are 11-20 classes. Defaults to "tab20".
        cmap_many_classes (str, optional):
            Colormap to use if there are more than 20 classes. Defaults to "viridis".

    Returns:
        dict: Containing the cmap, vmin/vmax, and whether the colormap is discrete
    """
    # This could be written in fewer lines of code but I kept it intentionally explicit

    if IDs_to_labels is None:
        # No IDs_to_labels means it's continous
        cmap = cmap_continous
        vmin = None
        vmax = None
        discrete = False
    else:
        # Otherwise, we can determine the max class ID
        max_ID = np.max(list(IDs_to_labels.keys()))

        if max_ID < 10:
            # 10 or fewer discrete classes
            cmap = cmap_10_classes
            vmin = -0.5
            vmax = 9.5
            discrete = True
        elif max_ID < 20:
            # 11-20 discrete classes
            cmap = cmap_20_classes
            vmin = -0.5
            vmax = 19.5
            discrete = True
        else:
            # More than 20 classes. There are no good discrete colormaps for this, so we generally
            # fall back on displaying it with a continous colormap
            cmap = cmap_many_classes
            vmin = None
            vmax = None
            discrete = False

    return {"cmap": cmap, "vmin": vmin, "vmax": vmax, "discrete": discrete}


def create_composite(
    RGB_image: np.ndarray,
    label_image: np.ndarray,
    label_blending_weight: float = 0.5,
    IDs_to_labels: typing.Union[None, dict] = None,
    grayscale_RGB_overlay: bool = True,
):
    """Create a three-panel composite with an RGB image and a label

    Args:
        RGB_image (np.ndarray):
            (h, w, 3) rgb image to be used directly as one panel
        label_image (np.ndarray):
            (h, w) image containing either integer labels or float scalars. Will be colormapped
            prior to display.
        label_blending_weight (float, optional):
            Opacity for the label in the blended composite. Defaults to 0.5.
        IDs_to_labels (typing.Union[None, dict], optional):
            Mapping from integer IDs to string labels. Used to compute colormap. If None, a
            continous colormap is used. Defaults to None.
        grayscale_RGB_overlay (bool):
            Convert the RGB image to grayscale in the overlay. Default is True.

    Raises:
        ValueError: If the RGB image cannot be interpreted as such

    Returns:
        np.ndarray: (h, 3*w, 3) horizontally composited image
    """
    if RGB_image.ndim != 3 or RGB_image.shape[2] != 3:
        raise ValueError("Invalid RGB error")

    if RGB_image.dtype == np.uint8:
        # Rescale to float range and implicitly cast
        RGB_image = RGB_image / 255

    vis_options = get_vis_options_from_IDs_to_labels(IDs_to_labels)
    if label_image.dtype == np.uint8 and not vis_options["discrete"]:
        # Rescale to float range and implicitly cast
        label_image = label_image / 255

    if not (label_image.ndim == 3 and label_image.shape[2] == 3):
        # If it's a one channel image make it not have a channel dim
        label_image = np.squeeze(label_image)

        cmap = plt.get_cmap(vis_options["cmap"])
        null_mask = np.logical_or(
            label_image == NULL_TEXTURE_INT_VALUE,
            np.logical_not(np.isfinite(label_image)),
        )
        if not vis_options["discrete"]:
            # Shift
            label_image = label_image - np.nanmin(label_image)
            # Find the max value that's not the null vlaue
            valid_pixels = label_image[np.logical_not(null_mask)]
            if valid_pixels.size > 0:
                # TODO this might have to be changed to nanmax in the future
                max_value = np.max(valid_pixels)
                # Scale
                label_image = label_image / max_value

        # Perform the colormapping
        label_image = cmap(label_image)[..., :3]
        # Mask invalid values
        label_image[null_mask] = 0

    # Create a blended image
    if grayscale_RGB_overlay:
        RGB_for_composite = np.tile(
            np.mean(RGB_image, axis=2, keepdims=True), (1, 1, 3)
        )
    else:
        RGB_for_composite = RGB_image
    overlay = ((1 - label_blending_weight) * RGB_for_composite) + (
        label_blending_weight * label_image
    )
    # Concatenate the images horizonally
    composite = np.concatenate((label_image, RGB_image, overlay), axis=1)
    # Cast to np.uint8 for saving
    composite = (composite * 255).astype(np.uint8)
    return composite


def read_img_npy(filename):
    try:
        return imread(filename)
    except:
        pass

    try:
        return np.load(filename)
    except:
        pass


def show_segmentation_labels(
    label_folder: PATH_TYPE,
    image_folder: PATH_TYPE,
    savefolder: typing.Union[None, PATH_TYPE] = None,
    num_show: int = 10,
    label_suffix: str = ".png",
    image_suffix: str = ".JPG",
    IDs_to_labels: typing.Optional[dict] = None,
) -> None:
    """
    Visualize and optionally save composite images showing segmentation labels overlaid
    on their corresponding images.

    Args:
        label_folder (PATH_TYPE): Path to the folder containing label images.
        image_folder (PATH_TYPE): Path to the folder containing original images.
        savefolder (PATH_TYPE, optional): If provided, composites are saved here;
            otherwise, they are displayed using pyplot. Defaults to None.
        num_show (int): Number of samples to show or save. Defaults to 10.
        label_suffix (str): Suffix for label image files. Defaults to ".png".
        image_suffix (str): Suffix for image files. Defaults to ".JPG".
        IDs_to_labels (dict, optional): Mapping from label IDs to class names.
            If None, will attempt to load from label_folder. Defaults to None.
    """
    # Find all label files in the label_folder and shuffle them
    rendered_files = list(Path(label_folder).rglob("*" + label_suffix))
    np.random.shuffle(rendered_files)

    # Ensure the save folder exists if saving output
    if savefolder is not None:
        ensure_folder(savefolder)

    # Attempt to load IDs_to_labels from a JSON file if not provided
    if (
        IDs_to_labels is None
        and (IDs_to_labels_file := Path(label_folder, "IDs_to_labels.json")).exists()
    ):
        with open(IDs_to_labels_file, "r") as infile:
            IDs_to_labels = json.load(infile)
            IDs_to_labels = {int(k): v for k, v in IDs_to_labels.items()}
    # Iterate through a subset of label files and create composites
    for i, rendered_file in tqdm(
        enumerate(rendered_files[:num_show]),
        desc="Showing segmentation labels",
        total=min(len(rendered_files), num_show),
    ):
        # Find the corresponding image file by matching the label path to an image path.
        # Assumes that image_folder and label_folder have the same subdirectory structure
        # and file stems for corresponding images/labels.
        image_file = Path(
            image_folder, rendered_file.relative_to(label_folder)
        ).with_suffix(image_suffix)

        # Read the image and label data
        image = imread(image_file)
        render = read_img_npy(rendered_file)
        # Create a composite visualization
        composite = create_composite(image, render, IDs_to_labels=IDs_to_labels)

        if savefolder is None:
            # Display the composite
            plt.imshow(composite)
            plt.show()
        else:
            # Save the composite to the output folder
            output_file = Path(savefolder, f"rendered_label_{i:03}.png")
            imwrite(output_file, composite)


def visualize_intersections_in_pyvista(
    plotter: pv.Plotter,
    ray_starts: np.ndarray,
    ray_ends: np.ndarray,
    community_IDs: np.ndarray,
    community_points: np.ndarray,
) -> None:
    """
    Visualize the given grouped rays and detected points in a pyvista plotter.

    Arguments:
        plotter (pv.Plotter):
            Existing pyvista plotter to add intersection lines/points to
        ray_starts ((N, 3) np.ndarray):
            The 3D locations of the starting points of N rays
        ray_ends ((N, 3) np.ndarray):
            The 3D locations of the ending points of N rays
        community_IDs ((N,) np.ndarray):
            The IDs for groups of rays. For example, if there are 10 rays with 5 in group 0
            and 5 in group 1, this would be [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        community_points ((M, 3) np.ndarray):
            One 3D point per community, indicating the center of the grouped rays. In the
            example above, this would be an (2, 3) array with 2 points.
    """

    # Interweave the points as line_segments_from_points desires
    interwoven = np.empty((2 * len(ray_starts), 3), dtype=ray_starts.dtype)
    interwoven[0::2] = ray_starts
    interwoven[1::2] = ray_ends

    # Show the line segments
    lines_mesh = pv.line_segments_from_points(interwoven)
    plotter.add_mesh(
        lines_mesh,
        scalars=community_IDs,
        label="Rays, colored by community ID",
    )

    # Show the triangulated communtities as red spheres
    detected_points = pv.PolyData(community_points)
    plotter.add_points(
        detected_points,
        color="r",
        render_points_as_spheres=True,
        point_size=10,
        label="Triangulated locations",
    )
    plotter.add_legend()


def visualize_intersections_as_mesh(
    ray_starts: np.ndarray,
    ray_ends: np.ndarray,
    community_IDs: np.ndarray,
    community_points: np.ndarray,
    out_dir: PATH_TYPE,
    batch: int = 250,
    cube_side_len: float = 0.2,
) -> None:
    """
    Arguments:
        ray_starts ((N, 3) np.ndarray):
            The 3D locations of the starting points of N rays
        ray_ends ((N, 3) np.ndarray):
            The 3D locations of the ending points of N rays
        community_IDs ((N,) np.ndarray):
            The IDs for groups of rays. For example, if there are 10 rays with 5 in group 0
            and 5 in group 1, this would be [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        community_points ((M, 3) np.ndarray):
            One 3D point per community, indicating the center of the grouped rays. In the
            example above, this would be an (2, 3) array with 2 points.
        out_dir (PATH_TYPE):
        batch (int):
            Defaults to 250.
        cube_side_len (float):
            Defaults to 0.2

    Saves:
        out_dir / rays.ply
        out_dir / points.ply
    """

    # Short-circuit on empty
    if len(ray_starts) == 0:
        return

    # Enforce Path type
    out_dir = Path(out_dir)

    # Split the communities into an (unforunately limited) color set
    norm = Normalize(vmin=np.nanmin(community_IDs), vmax=np.nanmax(community_IDs))
    cmap = colormaps["tab20"]

    # Build up cylinders in batches
    n_batches = int(np.ceil(len(community_IDs) / batch))
    cylinder_polydata = None
    for i in tqdm(range(n_batches), desc="Building cylinders"):
        islice = slice(i * batch, min((i + 1) * batch, len(community_IDs)))
        batched = merge_cylinders(
            starts=ray_starts[islice],
            ends=ray_ends[islice],
            community_IDs=community_IDs[islice],
            cmap=cmap,
            norm=norm,
        )
        # Merge with previous cylinders
        if cylinder_polydata is None:
            cylinder_polydata = batched
        else:
            cylinder_polydata = cylinder_polydata.merge(batched)

    if cylinder_polydata is not None:
        path = out_dir / "rays.ply"
        print(f"Saving visualized cylinders to {path}")
        cylinder_polydata.save(path, texture="RGB")

    # The cube merging is much less costly than the cylinders, and thus far hasn't
    # required batching
    cube_polydata = None
    for community_ID, point in enumerate(
        tqdm(community_points, desc="Building points")
    ):
        # Build a cube and set the color
        cube = pv.Cube(
            center=point,
            x_length=cube_side_len,
            y_length=cube_side_len,
            z_length=cube_side_len,
        )
        color = (np.array(cmap(norm(community_ID)))[:3] * 255).astype(np.uint8)
        cube.point_data["RGB"] = np.tile(color, (cube.n_points, 1))
        # Merge with previous cubes
        if cube_polydata is None:
            cube_polydata = cube
        else:
            cube_polydata = cube_polydata.merge(cube)

    if cube_polydata is not None:
        path = out_dir / "points.ply"
        print(f"Saving visualized cubes to {path}")
        cube_polydata.save(path, texture="RGB")


def merge_cylinders(
    starts: np.ndarray,
    ends: np.ndarray,
    community_IDs: np.ndarray,
    cmap: ListedColormap,
    norm: Normalize,
) -> pv.PolyData:
    """
    Create and merge a set of cylinders one by one.

    Arguments:
        ray_starts ((N, 3) np.ndarray):
            The 3D locations of the starting points of N rays
        ray_ends ((N, 3) np.ndarray):
            The 3D locations of the ending points of N rays
        community_IDs ((N,) np.ndarray):
            The IDs for groups of rays. For example, if there are 10 rays with 5 in group 0
            and 5 in group 1, this would be [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        cmap (matplotlib.colors.ListedColormap):
            Colormap to use on the normalized community IDs
        norm (matplotlib.colors.Normalize):
            Normalization function that spans from min to max of the community IDs

    Returns: pv.Polydata mesh representing the given rays with cylinders.
    """

    polydata = None
    for start, end, community_ID in zip(starts, ends, community_IDs):

        # Build a cylinder
        center = (start + end) / 2
        direction = end - start
        height = np.linalg.norm(direction)
        if height == 0:
            continue
        direction = direction / height
        # Some of the chosen parameters (low resolution, no capping) are to reduce
        # polygon faces when dealing with large numbers of cylinders
        cyl = pv.Cylinder(
            center=center,
            direction=direction,
            radius=0.05,
            height=height,
            resolution=3,
            capping=False,
        )

        # Color the cylinder
        color = (np.array(cmap(norm(community_ID)))[:3] * 255).astype(np.uint8)
        cyl["scalars"] = np.full(cyl.n_points, community_ID)
        cyl.point_data["RGB"] = np.tile(color, (cyl.n_points, 1))

        # And merge it into the scene
        if polydata is None:
            polydata = cyl
        else:
            polydata = polydata.merge(cyl)

    return polydata
