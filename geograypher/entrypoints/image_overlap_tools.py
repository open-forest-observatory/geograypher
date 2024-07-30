from typing import Union

import numpy as np
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
    downsample_target: float = 1,
    compute_projection: bool = False,
    compute_minimal_set: bool = False,
    vis: bool = False,
    projections_filename: Union[PATH_TYPE, None] = None,
    selected_images_mask_filename: Union[PATH_TYPE, None] = None,
    min_observations_to_be_included: int = 1,
):
    if compute_projection:
        mesh = TexturedPhotogrammetryMeshIndexPredictions(
            mesh=mesh_file,
            downsample_target=downsample_target,
            transform_filename=cameras_file,
        )
        camera_set = MetashapeCameraSet(camera_file=cameras_file, image_folder="")
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

        problem = SetCover(projection_matrix, set_costs)
        # TODO see what is contained in the solution
        solution_cost, time_used = problem.SolveSCP()
        print(
            f"The solution cost is {solution_cost} and solving took {time_used} minutes"
        )

        # Save out the mask representing the selected images
        selected_images = problem.s
        ensure_containing_folder(selected_images_mask_filename)
        np.save(selected_images_mask_filename, selected_images)
