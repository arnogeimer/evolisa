import copy
import random
from typing import List

from PIL import Image, ImageDraw, ImageFilter

from utils.dna import DNA
from utils.polygon import Polygon


class DNA(object):
    """DNA"""

    def __init__(self, img_size: tuple, img_path: str, polygons: List[Polygon] = []):
        self.img_size: tuple = img_size
        self.img_path: str = img_path
        self.polygons: List[Polygon] = polygons
        self.generation: int = 0

    def __str__(self):
        return self.__unicode__().encode("utf-8")

    def __unicode__(self):
        return "{}".format(self.polygons)

    def print_polygons(self):
        """
        debug function to print all DNA polygon info.
        """
        for polygon in self.polygons:
            print(polygon)

    def draw(self, background=(0, 0, 0, 0), save=False, name=None):
        """
        paint all DNA polygons onto an Image and show it.
        """
        size = self.img_size
        canvas = Image.new("RGBA", size, background)
        draw = Image.new("RGBA", size)

        for polygon in self.polygons:
            layer = Image.new("RGBA", size)
            draw = ImageDraw.Draw(layer, "RGBA")
            colour = polygon.colour
            points = polygon.points
            draw.polygon(points, fill=colour, outline=None)
            canvas = Image.alpha_composite(canvas, layer)

        background = Image.new("RGB", canvas.size, (255, 255, 255))
        background.paste(canvas, mask=canvas.split()[3])  # use alpha as mask
        canvas = background

        if save:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=3))
            canvas.save(f"./img_out/{name}.png")
        return canvas

    def mutate_dna(
        self,
        num_mutations: int = 1,
        offsetcol: int = 10,
        offsetx: int = 10,
        offsety: int = 10,
        frozen_polygons: tuple[int, int] = (0, 0),
    ):
        """
        mutate the dna.
        """
        # pick a random polygon
        polygons = copy.deepcopy(self.polygons)
        if frozen_polygons[1] > len(polygons):
            frozen_polygons = (frozen_polygons[0], len(polygons))
        rand_idx = random.choice(
            [
                i
                for i in range(0, len(polygons))
                if i not in range(frozen_polygons[0], frozen_polygons[1])
            ]
        )
        random_polygon = polygons[rand_idx]
        for _ in range(num_mutations):
            rand = random.random()
            if rand <= 0.35:
                random_polygon.mutate_colour(offset=offsetcol)
            elif rand <= 0.7:
                random_polygon.mutate_point(
                    self.img_size, offsetx=offsetx, offsety=offsety
                )
            elif rand <= 0.85:
                random_polygon.add_vertex(self.img_size)
            elif rand <= 0.95:
                random_polygon.remove_vertex()
            else:
                random_polygon.move_polygon(
                    self.img_size, offsetx=offsetx, offsety=offsety
                )
        return DNA(self.img_size, self.img_path, polygons)


def _generate_point(width, height):
    """
    generate random (x,y) coordinates.
    """
    x = random.randrange(0, width, 1)
    y = random.randrange(0, height, 1)
    return (x, y)


def _generate_colour():
    """
    generate random (r,g,b,a) colour.
    """
    red = random.randrange(0, 256)
    green = random.randrange(0, 256)
    blue = random.randrange(0, 256)
    alpha = random.randrange(0, 256)
    return (red, green, blue, alpha)


def generate_dna(
    img_size: tuple,
    img_path: str,
    dna_size: int = 50,
    fixed_colour: bool = False,
) -> DNA:
    """
    generate dna string consisting of polygons.
    """
    dna = None
    polygons = []
    (width, height) = img_size

    for _ in range(dna_size):
        nr_of_points = random.randrange(3, 8)
        points = []
        for _ in range(nr_of_points):
            # generate a point (x,y) in 2D space and append it to points.
            point = _generate_point(width, height)
            points.append(point)

        # generate colour (r,g,b,a) for polygon
        # colour = COLOUR_BLACK if fixed_colour else generate_colour()
        colour = (255, 255, 255, 255) if fixed_colour else _generate_colour()
        polygon = Polygon(colour, points)
        polygons.append(polygon)

    dna = DNA(img_size, img_path, polygons)
    return dna
