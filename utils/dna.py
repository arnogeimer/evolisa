import copy
import random
from typing import List

from utils.polygon import Polygon
from utils.utils import oit_composite, load_image, image_to_numpy, fitness, draw_polygon_rgb
import tqdm

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



class DNA(object):
    """DNA"""

    def __init__(self, img_path: str, fixed_polygons: List[Polygon] = [], trainable_polygons: List[Polygon] = []):
        self.original_img_path: str = img_path
        self.original_img_size: tuple = load_image(self.original_img_path).size
        self.fixed_polygons: List[Polygon] = fixed_polygons
        self.trainable_polygons: List[Polygon] = trainable_polygons
        self.fitness = None

    def __str__(self):
        return self.__unicode__().encode("utf-8")

    def __unicode__(self):
        return "{}".format(self.polygons)

    def print_polygons(self):
        """
        debug function to print all DNA polygon info.
        """
        for polygon in self.fixed_polygons + self.trainable_polygons:
            print(polygon)

    def generate_dna(
        self,
        dna_size: int = 50,
        fixed_colour: bool = False,
    ) -> None:
        """
        generate dna string consisting of polygons.
        """
        (width, height) = self.original_img_size
        dna = []
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
            dna.append(polygon)
        self.trainable_polygons = dna


    def draw(self, draw_full: bool = True):
        """
        paint all DNA polygons onto an Image and show it.
        """
        if draw_full:
            return oit_composite(polygons = self.fixed_polygons + self.trainable_polygons, size = self.original_img_size)
        else:
            return oit_composite(polygons = self.trainable_polygons, size = self.original_img_size)

    def _mutate(
        self,
        offsetcol: int = 10,
        offsetx: int = 10,
        offsety: int = 10,
    ) -> List[Polygon]:
        """
        mutate the dna.
        """
        # pick a random polygon
        polygons = copy.deepcopy(self.trainable_polygons)
        rand = random.random()
        if rand <= 0.5:
            random.choice(polygons).mutate_colour(offset=offsetcol)
        elif rand <= 1:
            random.choice(polygons).mutate_point(
                self.original_img_size, offsetx=offsetx, offsety=offsety
                )
            '''elif rand <= 0.85:
                random_polygon.add_vertex(self.img_size)
            elif rand <= 0.95:
                random_polygon.remove_vertex()
            else:
                random_polygon.move_polygon(
                    self.img_size, offsetx=offsetx, offsety=offsety
                )'''
        return polygons

    def evolve(
        self,
        num_epochs: int,
        offsetcol: int = 10,
        offsetx: int = 10,
        offsety: int = 10,
    ):
        original_numpy = image_to_numpy(load_image(self.original_img_path))
        num_changes = 0
        for _ in tqdm.trange(num_epochs):
            dna_child = self._mutate(
                offsetcol=offsetcol,
                offsetx=offsetx,
                offsety=offsety,
            )
            fitness_parent = fitness(
                original_numpy, image_to_numpy(draw_polygon_rgb(size = self.original_img_size, polygons = self.fixed_polygons + self.trainable_polygons))
            )
            fitness_child = fitness(
                original_numpy, image_to_numpy(draw_polygon_rgb(polygons = self.fixed_polygons + dna_child, size = self.original_img_size).convert("RGB"))
            )
            if fitness_child < fitness_parent:
                self.trainable_polygons = dna_child
                fitness_parent = fitness_child
                num_changes += 1
        self.fitness = fitness_parent
        random.shuffle(self.fixed_polygons)
        random.shuffle(self.trainable_polygons)
        #print(f"{self.name} made {num_changes} improvements in round {round}.")
        return num_changes

    def copy_polygons(self, num_polygons: int = 50) -> None:
        """Copy some polygons and halve their and their parents' alpha to add more DNA to the client."""
        duplicate_indices = sorted(
            random.sample(range(len(self.trainable_polygons)), num_polygons)
        )
        # Halve alpha for duplicate polygons
        for i in duplicate_indices:
            self.trainable_polygons[i].halve_alpha()
        duplicate_frozen_polygons = copy.deepcopy([self.trainable_polygons[i] for i in duplicate_indices])
        self.fixed_polygons += duplicate_frozen_polygons

    def save_current_draw(self, name: str):
        image = self.draw(draw_full=True)
        image.save(f"./img_out/{name}.png")
