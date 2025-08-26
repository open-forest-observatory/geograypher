import argparse
from typing import Optional, Union

import numpy as np
import pyproj
import pyvista as pv
from scipy.sparse import load_npz, save_npz
from SetCoverPy.setcover import SetCover

from geograypher.cameras import MetashapeCameraSet, SegmentorPhotogrammetryCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshIndexPredictions
from geograypher.predictors.derived_segmentors import ImageIDSegmentor
from geograypher.utils.files import ensure_containing_folder


def determine_minimum_overlapping_images(
    mesh_file: PATH_TYPE,
    cameras_file: PATH_TYPE,
    mesh_CRS: pyproj.CRS,
    image_folder: PATH_TYPE = "",
    ROI: Optional[PATH_TYPE] = None,
    ROI_buffer_meters: float = 0.0,
    compute_projection: bool = False,
    compute_minimal_set: bool = False,
    save_selected_images: bool = False,
    projections_filename: Union[PATH_TYPE, None] = None,
    selected_images_mask_filename: Union[PATH_TYPE, None] = None,
    selected_images_save_folder: Union[PATH_TYPE, None] = None,
    downsample_target: float = 1,
    min_observations_to_be_included: int = 1,
    vis: bool = False,
):
    """Determine a subset of images that together observe (or nearly observe) the entire scene.

    Args:
        mesh_file (PATH_TYPE):
            Path to the mesh file exported by Metashape
        cameras_file (PATH_TYPE):
            Path to the cameras file exported by Metashape
        mesh_CRS (pyproj.CRS):
            The CRS to interpret the mesh in.
        image_folder (PATH_TYPE, optional):
            Path to the image folder used to generate the project. Only required if
            save_selected_images=True. Defaults to "".
        ROI (PATH, optional):
            The region of interest to include. Applied to both the mesh and the cameas. Defaults to
            None.
        ROI_buffer_meters (typing.Optional[float]):
            The distance in meters to include around the ROI for the mesh and the cameras. Defaults
            to 0.0.
        compute_projection (bool, optional):
            Compute which images project to which faces. Defaults to False.
        compute_minimal_set (bool, optional):
            Compute the minimal set of images required to cover the whole scene. Requires that the
            projections_filename points to a valid projections file. Defaults to False.
        save_selected_images (bool, optional):
            Save the selected images out to selected_images_save_folder. Requires that
            selected_images_mask_filename contain a mask indicating which images should be
            included. Defaults to False.
        projections_filename (Union[PATH_TYPE, None], optional):
            Where to save or load from the projections to faces. Will have a .npz extension.
            Defaults to None.
        selected_images_mask_filename (Union[PATH_TYPE, None], optional):
            Where to save or load from the selected subset of images as a binary mask. Will have
            a .npy extension. Defaults to None.
        selected_images_save_folder (Union[PATH_TYPE, None], optional):
            Where to save the selected folder of images. Defaults to None.
        downsample_target (float, optional):
            Downsample the mesh to this fraction of the original faces. Defaults to 1.
        min_observations_to_be_included (int, optional):
            Ensure that a selected camera observes all faces that are observed by at least this many
            cameras in the original scene. Setting to a higher value allows you to avoid including
            cameras only because they are the sole camera to observe a few faces. Defaults to 1.
        vis (bool, optional):
            Show intermediate results. Note that this will cause the process to hang until the
            visualization is closed. Defaults to False.
    """
    if compute_projection:
        # Load the mesh and optionally downsample it
        mesh = TexturedPhotogrammetryMeshIndexPredictions(
            mesh=mesh_file,
            input_CRS=mesh_CRS,
            downsample_target=downsample_target,
            ROI=ROI,
            ROI_buffer_meters=ROI_buffer_meters,
        )
        # Load the camera set
        camera_set = MetashapeCameraSet(
            camera_file=cameras_file, image_folder=image_folder
        )
        # If the ROI is provided, subset the cameras to it
        if ROI is not None:
            camera_set = camera_set.get_subset_ROI(ROI, buffer_radius=ROI_buffer_meters)
        # Visualize the inputs if requested
        if vis:
            mesh.vis(camera_set=camera_set, frustum_scale=15)

        # Get all the image filenames
        image_filenames = camera_set.get_image_filename(index=None, absolute=True)
        # Create a segmentor that returns an image with all pixels set to the index of the image
        # within the set of images
        segmentor = ImageIDSegmentor(image_filenames=image_filenames)
        # Wrap the camera set in the segmentor
        segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
            base_camera_set=camera_set, segmentor=segmentor
        )

        # Project the images to the mesh
        _, additional_info = mesh.aggregate_projected_images(
            cameras=segmentor_camera_set, n_classes=len(camera_set)
        )
        # Extract the summed (unnormalized projections)
        summed_projections = additional_info["summed_projections"]
        # Save the projections
        ensure_containing_folder(projections_filename)
        save_npz(projections_filename, summed_projections)

        if vis:
            # Compute the number of projections per face
            counts = np.squeeze(np.asarray(np.sum(summed_projections, axis=1)))
            # Compute the index of the camera that projects to a face. If multiple do, the one with
            # the lowest index will be reported
            projected_camera = np.asarray(np.argmax(summed_projections, axis=1)).astype(
                float
            )
            # Mask out faces with no projections
            projected_camera[counts == 0] = np.nan

            print("Showing the number of projections per face")
            mesh.vis(
                vis_scalars=counts,
                plotter_kwargs={"title": "Number of cameras projecting to each face"},
            )
            mesh.vis(
                vis_scalars=projected_camera,
                plotter_kwargs={
                    "title": "Index of projected camera, lowest in case of multiple"
                },
            )

    if compute_minimal_set:
        # Load the projections
        projection_matrix = load_npz(projections_filename).astype(bool)
        # Determine how many images project to each face
        projected_images_per_face = np.squeeze(
            np.asarray(np.sum(projection_matrix, axis=1))
        )
        # Filter out any faces that don't have at least the threshold observation. This stops us
        # from requiring an image just because it observes a few faces that no other image does
        valid_rows = np.squeeze(
            projected_images_per_face >= min_observations_to_be_included
        )
        projection_matrix = projection_matrix[valid_rows, :]
        # Convert to a dense matrix
        # TODO see if we can avoid this step since it takes a fair bit of time and memory
        projection_matrix = np.asarray(projection_matrix.todense())

        # Define the costs for including each image as unit
        set_costs = np.ones(projection_matrix.shape[1]).astype(int)

        # Set up the set cover problem
        problem = SetCover(projection_matrix, set_costs)
        # Solve the problem, this can be slow
        print("Solving the set cover problem to determine the minimum set of cameras")
        solution_cost, time_used = problem.SolveSCP()
        print(
            f"The solution cost is {solution_cost} and solving took {time_used} minutes"
        )

        # Save out the mask representing the selected images
        selected_images = problem.s
        ensure_containing_folder(selected_images_mask_filename)
        np.save(selected_images_mask_filename, selected_images)

    if save_selected_images:
        # Load the camera set
        camera_set = MetashapeCameraSet(
            camera_file=cameras_file, image_folder=image_folder
        )
        # If the ROI is provided, subset the cameras to it
        if ROI is not None:
            camera_set = camera_set.get_subset_ROI(ROI, buffer_radius=ROI_buffer_meters)
        # Load the mask identifying the selected cameras
        cameras_mask = np.load(selected_images_mask_filename)
        # Convert it from a boolean mask to indices
        cameras_indices = np.where(cameras_mask)[0]
        # Take the corresponding set of cameras
        subset_cameras_set = camera_set.get_subset_cameras(cameras_indices)
        # Save out the images corresponding to the selected subset
        subset_cameras_set.save_images(output_folder=selected_images_save_folder)

        # Show the selected cameras bigger
        if vis:
            plotter = pv.Plotter()
            # TODO update these constants to be more robust
            # The issue is they are in the internal coordinate system of the camera set
            # and the scale changes between projects. Ideally, we'd want to specify it in meters.
            camera_set.vis(plotter=plotter, frustum_scale=0.4)
            subset_cameras_set.vis(plotter=plotter, frustum_scale=0.8)
            print("Showing the cameras, with selected ones larger.")
            plotter.show()


def parse_args():
    description = (
        "This script determines a minimum set of images that fully observe the mesh."
        + "The command line arguments are directly passed to"
        + "geograypher.entrypoints.workflow_functions.determine_minimum_overlapping_images "
        + "which has the following documentation:\n\n"
        + determine_minimum_overlapping_images.__doc__
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--cameras-file", required=True)
    parser.add_argument("--mesh-CRS", required=True, type=int)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--ROI")
    parser.add_argument("--ROI-buffer-meters", type=float, default=0.0)
    parser.add_argument("--compute-projection", action="store_true")
    parser.add_argument("--compute-minimal-set", action="store_true")
    parser.add_argument("--save-selected-images", action="store_true")
    parser.add_argument("--projections-filename")
    parser.add_argument("--selected-images-mask-filename")
    parser.add_argument("--selected-images-save-folder")
    parser.add_argument("--downsample-target", default=1.0, type=float)
    parser.add_argument("--min-observations-to-be-included", default=1, type=float)
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass command line args to determine_minimum_overlapping_images
    determine_minimum_overlapping_images(**args.__dict__)
