import numpy as np
import pyvista as pv
from scipy.linalg import lstsq


def compute_approximate_ray_intersection(
    A: np.array, a: np.array, B: np.array, b: np.array, vis=False
):
    # https://palitri.com/vault/stuff/maths/Rays%18closest%20point.pdf
    c = B - A

    aa = np.dot(a, a)
    ab = np.dot(a, b)
    ac = np.dot(a, c)
    bb = np.dot(b, b)
    bc = np.dot(b, c)

    denominator = aa * bb - ab * ab

    a_scaler = (-ab * bc + ac * bb) / denominator
    b_scaler = (ab * ac - bc * aa) / denominator

    D = A + a_scaler * a
    E = B + b_scaler * b

    dist = np.linalg.norm(D - E)
    # Check that it's in front of the camera
    # TODO I'm not sure if this will handle co-linear cases, probably should check for that
    valid = a_scaler > -1 and b_scaler > 0

    if vis:
        points = np.vstack([A, D, B, E, D, E])

        plotter = pv.Plotter()
        plotter.add_lines(points)
        plotter.add_points(points)
        plotter.background_color = "black"
        plotter.show()

    return dist, valid


def triangulate_rays_lstsq(starts, directions):
    # https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/ray.py#L284-L371

    # Ensure that we have unit vectors specifying the direction of the ray

    # Build a cross product matrix for each direction, this is the left side
    # The right side is the dot product between this and the start ray

    As = []
    bs = []
    # TODO build this all at once rather than iteratively
    for start, direction in zip(starts, directions):
        x, y, z = direction
        # https://math.stackexchange.com/questions/3764426/matrix-vector-multiplication-cross-product-problem
        cross_matrix = np.array(
            [
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0],
            ]
        )
        As.append(cross_matrix)
        b = cross_matrix @ start
        bs.append(b)

    A = np.concatenate(As, axis=0)
    b = np.concatenate(bs, axis=0)

    x, _, _, _ = np.linalg.lstsq(A, b)
    return x
