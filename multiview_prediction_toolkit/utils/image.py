import piexif
from PIL import Image


def get_GPS_exif(filename):
    im = Image.open(filename)
    exif_dict = piexif.load(im.info["exif"])
    lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
    lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
    lat = (
        lat[0][0] / lat[0][1]
        + lat[1][0] / (lat[1][1] * 60)
        + lat[2][0] / (lat[2][1] * 3600)
    )
    lon = (
        lon[0][0] / lon[0][1]
        + lon[1][0] / (lon[1][1] * 60)
        + lon[2][0] / (lon[2][1] * 3600)
    )
    # https://stackoverflow.com/questions/54196347/overwrite-gps-coordinates-in-image-exif-using-python-3-6
    return lon, lat
