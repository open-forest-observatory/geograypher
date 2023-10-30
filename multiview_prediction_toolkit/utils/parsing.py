import xml.etree.ElementTree as ET


def make_4x4_transform(
    self, rotation_str: str, translation_str: str, scale_str: str = "1"
):
    """Convenience function to make a 4x4 matrix from the string format used by Metashape

    Args:
        rotation_str (str): Row major with 9 entries
        translation_str (str): 3 entries
        scale_str (str, optional): single value. Defaults to "1".

    Returns:
        np.ndarray: (4, 4) A homogenous transform mapping from cam to world
    """
    rotation_np = np.fromstring(rotation_str, sep=" ")
    rotation_np = np.reshape(rotation_np, (3, 3))
    translation_np = np.fromstring(translation_str, sep=" ")
    scale = float(scale_str)
    transform = np.eye(4)
    transform[:3, :3] = rotation_np * scale
    transform[:3, 3] = translation_np
    return transform


def parse_transform_metashape(camera_file):
    tree = ET.parse(camera_file)
    root = tree.getroot()
    # first level
    chunk = root.find("chunk")
    # second level
    sensors = chunk.find("sensors")
    transform = chunk[1][0][0]
    rotation = transform[0].text
    translation = transform[1].text
    scale = transform[2].text
    local_to_epgs_4978_transform = self.make_4x4_transform(rotation, translation, scale)
    return local_to_epgs_4978_transform
