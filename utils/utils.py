import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import cupy as cp
from typing import List, Tuple
from utils.polygon import Polygon

def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img


def image_to_numpy(img: Image.Image) -> np.ndarray:
    """
    convert PIL Image to numpy ndarray.
    """
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def draw_polygon_rgba_to_numpy(size, points, colour):
    """Draw a polygon onto an RGBA pillow image."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.polygon(points, fill=colour)
    return np.asarray(img).astype(np.float32) / 255.0


def oit_composite(polygons, size):
    """
    polygons: list of numpy arrays (H,W,4) in pre-multiplied floats [0..1]
    size: (width, height)
    """
    W, H = size

    accum_rgb = np.zeros((H, W, 3), dtype=np.float32)
    accum_a = np.zeros((H, W), dtype=np.float32)

    for poly in polygons:

        poly = draw_polygon_rgba_to_numpy(size, poly.points, poly.colour)
        rgb = poly[..., :3]
        a = poly[..., 3]

        accum_rgb += rgb * a[..., None]
        accum_a += a
    # avoid division by zero
    eps = 1e-6
    final_rgb = accum_rgb / (accum_a[..., None] + eps)

    final_a = np.clip(accum_a, 0, 1)

    result = np.dstack([final_rgb, final_a])
    result = (result * 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGBA")

def draw_polygon_rgb(size: Tuple[int, int], polygons: List[Polygon]):
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = Image.new("RGBA", size)

    for polygon in polygons:
        layer = Image.new("RGBA", size)
        draw = ImageDraw.Draw(layer, "RGBA")
        colour = polygon.colour
        points = polygon.points
        draw.polygon(points, fill=colour, outline=None)
        canvas = Image.alpha_composite(canvas, layer)

    background = Image.new("RGB", canvas.size, (255, 255, 255))
    background.paste(canvas, mask=canvas.split()[3])  # use alpha as mask
    canvas = background

    return canvas


def fitness(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    fitness function determines how much alike an image (as ndarray) and DNA are.
    """
    return cp.linalg.norm(cp.array(img_1) - cp.array(img_2))


def generate_gif_from_output_images(name: str = "evolisa", num_rounds: int = 200):
    filenames = [
        f"./img_out/dna_{name}_round_{i}.png" for i in range(1, num_rounds + 1)
    ]
    images = [imageio.imread(file_path) for file_path in filenames]
    imageio.mimsave(f"./img_out/evolution_{name}.gif", images)


def combine_images():
    rounds = [0, 2, 4, 6, 10, 15]
    img_names = [
        [
            f"./img_out/ml1_combined_round_{i}.png",
            f"./img_out/ml2_combined_round_{i}.png",
            f"./img_out/ml_combined_round_{i}.png",
        ]
        for i in rounds
    ]
    img_names = [x for xs in img_names for x in xs]

    imgs = [Image.open(i) for i in img_names]
    w, h = imgs[0].size

    canvas = Image.new("RGBA", ((len(imgs) // 3) * w + w, 3 * h), color=(0, 0, 0))

    for i in range(1, len(imgs), 3):
        canvas.alpha_composite(imgs[i], ((i // 3) * w, 0))
        canvas.alpha_composite(imgs[i + 1], ((i // 3) * w, h))
        canvas.alpha_composite(imgs[i + 2], ((i // 3) * w, 2 * h))
    canvas.save("./img_out/grid.png")


class ParameterManager(object):
    def __init__(self, initial_offset: int = 100):
        self.offset = initial_offset
        self.num_changes_threshold = 5  # threshold for number of improvements per round
        self.num_changes_below_threshold = (
            0  # counter for rounds with less than threshold improvements
        )

    def get_offset(self) -> int:
        return int(max(1, self.offset))

    def should_copy_polygons(self, num_changes: int, threshold: int = 5) -> bool:
        """
        If we have had less than num_changes_threshold improvements for threshold rounds,
        return True to call add_dna.
        """
        if num_changes < self.num_changes_threshold:
            self.num_changes_below_threshold += 1
        else:
            self.num_changes_below_threshold = 0
        if self.num_changes_below_threshold > threshold:
            self.offset = max(1, int(self.offset * 0.8))
            self.num_changes_below_threshold = 0
            return True
        return False
