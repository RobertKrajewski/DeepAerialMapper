import pathlib
import pprint

import typer
import yaml
from loguru import logger
from PIL import Image


def extract_tif_meta(image_dir: str) -> None:
    """Extracts meta information from tif satellite images for map creation.

    Extracts from every ".tif" image in given directory information including location, scale and projection used and
    stores results in a "meta.yaml" file the in images dir.
    :param image_dir: Directory containing tif (not tiff!) images.
    """
    meta = {}
    image_dir = pathlib.Path(image_dir)
    for image_path in image_dir.glob("*.tif"):
        img = Image.open(str(image_path))
        img_exif = img.getexif()
        coords = img_exif[int(0x8482)][3:5]
        px2m = img_exif[int(0x830E)][:2]
        proj = img_exif[int(0x87B1)]
        width = img_exif[int(0x0100)]
        height = img_exif[int(0x0101)]

        if proj == "ETRS89 / UTM zone 32N|ETRS89|":
            proj = "epsg:25832"

        meta[image_path.stem] = {
            "origin": coords,
            "scale": px2m,
            "proj": proj,
            "width": width,
            "height": height,
        }
    logger.info(f"Extracted meta information:\n{pprint.pformat(meta)}")

    output_filepath = image_dir / "meta.yaml"
    with output_filepath.open("w") as f:
        yaml.safe_dump(meta, f, indent=2, sort_keys=False)
    logger.info(f"Stored results to {output_filepath}")


if __name__ == "__main__":
    typer.run(extract_tif_meta)
