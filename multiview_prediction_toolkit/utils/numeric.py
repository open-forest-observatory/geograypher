import numpy as np
import pyvista as pv


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
    print(aa, ab, ac, bb, bc)

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
