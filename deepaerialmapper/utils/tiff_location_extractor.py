import json
import pathlib
from PIL import Image, ExifTags


meta = {}

image_dir = "D:/ika/diss/hdmap/hdmap_data_aachen/01_qgis"
image_dir = pathlib.Path(image_dir)
for image_path in image_dir.glob("*.tif"):
    img = Image.open(str(image_path))
    img_exif = img.getexif()
    coords = img_exif[int(0x8482)][3:5]
    px2m = img_exif[int(0x830e)][:2]
    proj = img_exif[int(0x87b1)]
    width = img_exif[int(0x0100)]
    height = img_exif[int(0x0101)]

    if proj == "ETRS89 / UTM zone 32N|ETRS89|":
        proj = "epsg:25832"

    meta[image_path.stem] = {
        "origin": coords,
        "scale": px2m,
        "proj": proj,
        "width": width,
        "height": height
    }

with (image_dir / "meta.json").open("w") as f:
    json.dump(meta, f, indent=2)
