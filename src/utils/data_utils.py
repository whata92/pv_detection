import json
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from typing import List


def convert_json_to_img(img_path: str, json_path: str, out_path: str):
    with open(json_path, "r") as fp:
        annotation = json.load(fp)

    img_np = cv2.imread(img_path)
    h, w, _ = img_np.shape
    mask = Image.new("L", (w, h))
    draw = ImageDraw.Draw(mask)
    
    for target in annotation["shapes"]:
        points = target["points"]
        points = [tuple(point) for point in points]
        draw.polygon(points, fill=1)
    
    mask.save(out_path)


def crop_img(
        img_path: str, 
        out_dir: str, 
        crop_size: List[int] = [512, 512],
        overlap: List[int] = [0.2, 0.2],
        residual_crop: bool = True,
        suffix: str = None,
        is_annotation: bool = False
    ):

    os.makedirs(out_dir, exist_ok=True)
    if not is_annotation:
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype("uint8")

    if len(img.shape) == 3:
        (x_size, y_size, _) = img.shape
    elif len(img.shape) == 2:
        (x_size, y_size) = img.shape

    overlap_pix = [
        int(crop_size[0] * overlap[0]), 
        int(crop_size[1] * overlap[1])
    ]

    if residual_crop:
        last_x = x_size + 1 - overlap_pix[0]
        last_y = y_size + 1 - overlap_pix[1]
    else:
        last_x = x_size - crop_size[0] + 1
        last_y = y_size - crop_size[1] + 1
    
    for x_min in range(0, last_x, crop_size[0] - overlap_pix[0]):
            for y_min in range(0, last_y, crop_size[1] - overlap_pix[1]):
                x_min = min(x_min, x_size - crop_size[0])
                y_min = min(y_min, y_size - crop_size[1])
                x_max = x_min + crop_size[0]
                y_max = y_min + crop_size[1]

                img_stem, img_suffix = os.path.splitext(
                     os.path.basename(img_path)
                )

                if suffix:
                    img_suffix = suffix

                crop_img_name = (
                    f"{img_stem}_{x_min}_{y_min}_{x_max}_{y_max}{img_suffix}"
                )

                save_path = os.path.join(out_dir, crop_img_name)
                if not is_annotation:
                    cv2.imwrite(save_path, img[x_min:x_max, y_min:y_max, :])
                else:
                    cv2.imwrite(save_path, img[x_min:x_max, y_min:y_max])


if __name__ == "__main__":
    convert_json_to_img(
        "/workspace/data/raw/gmap/2023-02-10 211523.png",
        "/workspace/data/raw/gmap/2023-02-10 211523.json",
        "tmp.png"
    )
    