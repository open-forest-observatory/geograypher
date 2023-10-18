from multiview_prediction_toolkit.config import (
    DEFAULT_GEO_POINTS_FILE,
    DEFAULT_LOCAL_MESH,
)
from multiview_prediction_toolkit.meshes import (
    TexturedPhotogrammetryMesh,
    HeightAboveGroundPhotogrammertryMesh,
    TreeIDTexturedPhotogrammetryMesh,
    TreeSpeciesTexturedPhotogrammetryMesh,
)


EXAMPLE_NUM = 3

if EXAMPLE_NUM == 0:
    # Load a mesh at native resolution
    mesh = TexturedPhotogrammetryMesh(
        mesh_filename=DEFAULT_LOCAL_MESH, downsample_target=1.0
    )
    # And visualize it
    mesh.vis()

elif EXAMPLE_NUM == 1:
    # Load a downsampled version of the mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_filename=DEFAULT_LOCAL_MESH, downsample_target=0.02
    )
    # Note that the visualization has no color because this information was
    # lost during downsampling
    mesh.vis()

elif EXAMPLE_NUM == 2:
    # Load a mesh textured with geographic data from a point file
    geodata_mesh = TreeSpeciesTexturedPhotogrammetryMesh(
        DEFAULT_LOCAL_MESH,
        geo_point_file=DEFAULT_GEO_POINTS_FILE,
    )
    geodata_mesh.vis(mesh_kwargs={"cmap": "tab10", "vmin": 0, "vmax": 9})

elif EXAMPLE_NUM == 3:
    # Load a mesh textured with height above ground
    height_above_ground_mesh = HeightAboveGroundPhotogrammertryMesh(
        mesh_filename=DEFAULT_LOCAL_MESH, ground_height_threshold=None
    )
    height_above_ground_mesh.vis()

elif EXAMPLE_NUM == 4:
    # Load a mesh textured with tree species
    tree_ID_mesh = TreeIDTexturedPhotogrammetryMesh(
        mesh_filename=DEFAULT_LOCAL_MESH, collapse_IDs=False
    )
    tree_ID_mesh.vis()
