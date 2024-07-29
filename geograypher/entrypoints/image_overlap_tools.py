from typing import Union

from scipy.sparse import save_npz

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
):
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
