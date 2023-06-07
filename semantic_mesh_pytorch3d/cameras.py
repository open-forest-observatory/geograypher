import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from pyvista import demos
import csv

REFLECT_Z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
REFLECT_Y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class MetashapeCamera:
    def __init__(self, filename, transform, f, cx, cy, image_width, image_height):
        self.filename = filename
        self.transform = transform
        self.f = f
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height

    def rescale(self, scale: float):
        """
        Rescale transform in place
        """
        self.transform[:3, 3] = self.transform[:3, 3] * scale

    def get_pytorch3d_camera(self, device):
        import torch
        from pytorch3d.renderer import FoVPerspectiveCameras
        # Invert this because it's cam to world and we need world to cam
        transform_4x4_world_to_cam = np.linalg.inv(self.transform)

        R = torch.Tensor(np.array([transform_4x4_world_to_cam[:3, :3].T]))
        T = torch.Tensor([transform_4x4_world_to_cam[:3, 3]])

        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        return cameras 

    def vis(self, plotter: pv.Plotter, vis_scale=1):
        scaled_halfwidth = self.image_width / (self.f * 2)
        scaled_halfheight = self.image_height / (self.f * 2)
        scaled_cx = self.cx / self.f
        scaled_cy = self.cx / self.f

        vertices = np.vstack(
            (
                np.array(
                    [
                        [0, 0, 0],
                        [
                            scaled_cx + scaled_halfwidth,
                            scaled_cy + scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx + scaled_halfwidth,
                            scaled_cy - scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx - scaled_halfwidth,
                            scaled_cy - scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx - scaled_halfwidth,
                            scaled_cy + scaled_halfheight,
                            1,
                        ],
                    ]
                ).T
                * vis_scale,
                np.ones((1, 5)),
            )
        )
        projected_vertices = self.transform @ vertices
        rescaled_projected_vertices = projected_vertices[:3] / projected_vertices[3:]
        ## mesh faces
        faces = np.hstack(
            [
                [3, 0, 1, 2],
                [3, 0, 2, 3],
                [3, 0, 3, 4],
                [3, 0, 4, 1],
                [3, 1, 2, 3],
                [3, 3, 4, 1],
            ]  # square  # triangle  # triangle
        )
        frustum = pv.PolyData(rescaled_projected_vertices[:3].T, faces)
        frustum.triangulate()
        plotter.add_mesh(frustum)


class MetashapeCameraSet:
    def __init__(self, camera_file_base):
        (
            self.f,
            self.cx,
            self.cy,
            self.image_width,
            self.image_height,
        ) = self.parse_metashape_cam_file(camera_file=camera_file_base + ".xml")

        self.filenames, self.transforms = self.parse_txt_cam_file(
            camera_file_base + ".txt"
        )

        self.cameras = []

        for filename, transform in zip(self.filenames, self.transforms):
            new_camera = MetashapeCamera(
                filename,
                transform,
                self.f,
                self.cx,
                self.cy,
                self.image_width,
                self.image_height,
            )
            self.cameras.append(new_camera)

    def rescale(self, scale):
        for camera in self.cameras:
            camera.rescale(scale)

    def add_orientation_cube(self, plotter):
        ocube = demos.orientation_cube()
        plotter.add_mesh(ocube["cube"], show_edges=True)
        plotter.add_mesh(ocube["x_p"], color="blue")
        plotter.add_mesh(ocube["x_n"], color="blue")
        plotter.add_mesh(ocube["y_p"], color="green")
        plotter.add_mesh(ocube["y_n"], color="green")
        plotter.add_mesh(ocube["z_p"], color="red")
        plotter.add_mesh(ocube["z_n"], color="red")

    def vis(self, plotter: pv.Plotter):
        for camera in self.cameras:
            camera.vis(plotter)

    def make_4x4_transform(self, rotation_str, translation_str, scale_str):
        rotation_np = np.fromstring(rotation_str, sep=" ")
        rotation_np = np.reshape(rotation_np, (3, 3))
        translation_np = np.fromstring(translation_str, sep=" ")
        scale = float(scale_str)
        transform = np.eye(4)
        transform[:3, :3] = rotation_np
        transform[:3, 3] = translation_np
        transform[3, 3] = 1 / scale
        return transform

    def parse_metashape_cam_file(self, camera_file):
        """Parse intrinsic params from metashape xml output format. Note that the
        positions contained within this output look broken

        Args:
            camera_file (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Load the xml file
        # Taken from here https://rowelldionicio.com/parsing-xml-with-python-minidom/
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # first level
        chunk = root[0]
        # second level
        sensors = chunk[0]

        # sensors info
        sensor = sensors[0]
        calibration = sensor[4]
        f = float(calibration[1].text)
        cx = float(calibration[2].text)
        cy = float(calibration[3].text)

        width = float(calibration[0].get("width"))
        height = float(calibration[0].get("height"))
        return f, cx, cy, width, height

    def parse_txt_cam_file(self, camera_file):
        """Parse filenames and transforms from <TODO metashape output> format

        Args:
            camera_file (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(camera_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            # Discard headers
            for _ in range(2):
                next(csv_reader)

            filenames = []
            transforms = []
            for line in csv_reader:
                filenames.append(line[0])
                transform = np.eye(4)
                R = np.array(line[-9:]).astype(float)
                loc = np.array(line[1:4]).astype(float)
                # Unsure why the negation is needed
                transform[:3, :3] = -np.reshape(R, (3, 3))
                transform[:3, 3] = loc
                transforms.append(transform)

        return filenames, transforms


if __name__ == "__main__":
    plotter = pv.Plotter()
    plotter.add_axes()
    camera_set = MetashapeCameraSet(
        "/home/david/data/Safeforest_CMU_data_dvc/data/site_Gascola/04_27_23/collect_05/processed_02/metashape/left_camera_automated/exports/example-run-001_20230517T1827_camera"
    )
    camera_set.vis(plotter)
    mesh = pv.read(
        "/home/david/data/Safeforest_CMU_data_dvc/data/site_Gascola/04_27_23/collect_05/processed_02/metashape/left_camera_automated/exports/example-run-001_20230517T1827_low_res_local.ply"
    )
    mesh["RGB"] = mesh["RGB"] * 4
    plotter.add_mesh(mesh, rgb=True)
    camera_set.add_orientation_cube(plotter)
    plotter.show()
