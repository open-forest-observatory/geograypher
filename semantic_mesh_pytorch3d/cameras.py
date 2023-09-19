import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from pyvista import demos
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy.ma as ma


class MetashapeCamera:
    def __init__(self, filename, transform, f, cx, cy, image_width, image_height):
        """
        TODO define units
        """
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

    def check_valid_in_image(self, projected_verts, image_size):
        image_space_verts = projected_verts[:2] / projected_verts[2:3]
        image_space_verts = image_space_verts.T

        in_front_of_cam = projected_verts[2] > 0

        img_width, image_height = image_size
        valid_verts_bool = np.logical_and.reduce(
            (
                image_space_verts[:, 0] > 0,
                image_space_verts[:, 1] > 0,
                image_space_verts[:, 0] < img_width,
                image_space_verts[:, 1] < image_height,
                in_front_of_cam,
            )
        )
        valid_image_space_verts = image_space_verts[valid_verts_bool, :].astype(int)
        return valid_verts_bool, valid_image_space_verts

    def extract_colors(self, valid_bool, valid_locs, img):
        colors_per_vertex = np.zeros((valid_bool.shape[0], img.shape[2]))
        mask = np.ones((valid_bool.shape[0], img.shape[2])).astype(bool)
        valid_inds = np.where(valid_bool)[0]
        # Set the entries which are valid
        mask[valid_inds, :] = False

        i_locs = valid_locs[:, 1]
        j_locs = valid_locs[:, 0]
        valid_color_samples = img[i_locs, j_locs, :]
        colors_per_vertex[valid_inds, :] = valid_color_samples
        masked_color_per_vertex = ma.array(colors_per_vertex, mask=mask)
        return masked_color_per_vertex

    def splat_mesh_verts(self, mesh_verts, img):
        transform_4x4_world_to_cam = np.linalg.inv(self.transform)
        K = np.eye(3)
        K[0, 0] = self.f
        K[1, 1] = self.f
        K[0, 2] = self.cx + self.image_width / 2.0
        K[1, 2] = self.cy + self.image_height / 2.0
        homogenous_mesh_verts = np.concatenate(
            (mesh_verts, np.ones((mesh_verts.shape[0], 1))), axis=1
        ).T
        camera_frame_mesh_verts = transform_4x4_world_to_cam @ homogenous_mesh_verts
        camera_frame_mesh_verts = camera_frame_mesh_verts[:3]
        # TODO review terminology
        projected_verts = K @ camera_frame_mesh_verts
        valid_bool, valid_locs = self.check_valid_in_image(
            projected_verts=projected_verts,
            image_size=(self.image_width, self.image_height),
        )
        colors_per_vertex = self.extract_colors(valid_bool, valid_locs, img)
        return colors_per_vertex

    def get_pytorch3d_camera(self, device):
        import torch
        from pytorch3d.renderer import PerspectiveCameras

        # Invert this because it's cam to world and we need world to cam
        transform_4x4_world_to_cam = np.linalg.inv(self.transform)

        R = torch.Tensor(np.array([transform_4x4_world_to_cam[:3, :3].T]))
        T = torch.Tensor([transform_4x4_world_to_cam[:3, 3]])
        # See https://pytorch3d.org/docs/cameras
        image_size = ((self.image_height, self.image_width),)
        fcl_screen = (self.f,)
        prc_points_screen = (
            (self.cx + self.image_width / 2, self.cy + self.image_height / 2),
        )
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=fcl_screen,
            principal_point=prc_points_screen,
            device=device,
            in_ndc=False,
            image_size=image_size,
        )
        return cameras

    def vis(self, plotter: pv.Plotter, vis_scale=0.1, pixel_focal=False):
        if pixel_focal:
            scaled_halfwidth = self.image_width / (self.f * 2)
            scaled_halfheight = self.image_height / (self.f * 2)
        else:
            # TODO
            scaled_halfwidth = 0.7#self.image_width / (self.f * 2)
            scaled_halfheight = 0.5#self.image_height / (self.f * 2)
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
        colors = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
        ).astype(float)
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
        frustum["RGB"] = colors
        frustum.triangulate()
        plotter.add_mesh(frustum, scalars="RGB", rgb=True)


class MetashapeCameraSet:
    def __init__(self, camera_file_base):
        (
            self.f,
            self.cx,
            self.cy,
            self.image_width,
            self.image_height,
            self.filenames,
            self.transforms,
        ) = self.parse_metashape_cam_file(camera_file=camera_file_base + ".xml")

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
        #self.add_orientation_cube(plotter=plotter)

    def make_4x4_transform(self, rotation_str, translation_str, scale_str="1"):
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
        width = int(sensor[0].get("width"))
        height = int(sensor[0].get("height"))
        pixel_width = float(sensor[1].get("value"))
        pixel_height = float(sensor[2].get("value"))
        focal_length = float(sensor[3].get("value"))

        # Try to parse the filenames and transforms
        #component = chunk[1][0]
        #transform = component[0]
        #region = component[0]

        #transform_4x4 = self.make_4x4_transform(
        #    rotation_str=str(transform[0].text),
        #    translation_str=str(transform[1].text),
        #    scale_str=str(transform[2].text),
        #)

        #transform_4x4 = self.make_4x4_transform(
        #    rotation_str=str(region[2].text),
        #    translation_str=str(region[0].text),
        #)

        cameras = chunk[2]

        labels = []
        camera_transforms = []
        for camera in cameras:
            labels.append(camera.get("label"))
            camera_transforms.append(np.fromstring(camera[0].text, sep=" ").reshape(4,4))

        return focal_length, pixel_width, pixel_height, width, height, labels, camera_transforms

    def parse_txt_cam_file(self, camera_file):
        """Parse filenames and transforms from <TODO metashape output> format

        Args:
            camera_file (_type_): _description_

        Returns:
            _type_: _description_
        """
        REFLECT_Z = np.eye(3)
        REFLECT_Z[2, 2] = -1
        REFLECT_Y = np.eye(3)
        REFLECT_Y[1, 1] = -1

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
                R = np.reshape(R, (3, 3))
                # Reflect as suggested
                # https://github.com/EnricoAhlers/agi2nerf/blob/main/agi2nerf.py
                R = REFLECT_Z @ R
                R = REFLECT_Y @ R

                transform[:3, :3] = R
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
