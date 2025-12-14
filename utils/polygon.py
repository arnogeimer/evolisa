import copy
import random
from typing import List

import numpy as np


class Polygon(object):
    """Polygon"""

    def __init__(self, colour=None, points: List[tuple] = []):
        self.colour: tuple = colour
        self.points: List[tuple] = points

    def __str__(self):
        return f"{self.points}, {self.colour}"

    def __unicode__(self):
        return "{}, {}".format(self.points, self.colour)

    def halve_alpha(self):
        r, g, b, alpha = self.colour
        self.colour = (r, g, b, round(255 * (1 - np.sqrt(1 - (alpha / 255)))))

    def mutate_colour(self, offset: int = 50):
        """
        mutate the colour of the polygon.
        """
        idx = random.randrange(0, 4)
        colours = list(copy.deepcopy(self.colour))
        colours[idx] = max(
            0, min(255, colours[idx] + random.randrange(-offset, offset))
        )
        self.colour = tuple(colours)

    def mutate_point(self, img_size: tuple, offsetx: int = 50, offsety: int = 50):
        """
        mutate a random point of the polygon.
        """
        idx = random.randrange(0, len(self.points))
        points = list(copy.deepcopy(self.points))
        new_x, new_y = max(
            0, min(img_size[0], points[idx][0] + random.randrange(-offsetx, offsetx))
        ), max(
            0, min(img_size[1], points[idx][1] + random.randrange(-offsety, offsety))
        )
        points[idx] = (new_x, new_y)
        self.points = tuple(points)

    def add_vertex(self, img_size: tuple):
        """
        add a vertex to the polygon.
        """
        points = list(copy.deepcopy(self.points))
        new_point = (
            random.randrange(0, img_size[1], 1),
            random.randrange(0, img_size[1], 1),
        )
        points.append(new_point)
        self.points = tuple(points)

    def remove_vertex(self):
        """
        remove a vertex from the polygon.
        """
        idx = random.randrange(0, len(self.points))
        if len(self.points > 3):
            points = list(copy.deepcopy(self.points))
            points.pop(idx)
            self.points = tuple(points)

    def move_polygon(self, img_size: tuple, offsetx: int = 50, offsety: int = 50):
        """
        move the entire polygon.
        """
        points = list(copy.deepcopy(self.points))
        offset_x = random.randrange(-offsetx, offsetx)
        offset_y = random.randrange(-offsety, offsety)
        new_points = []
        for point in points:
            new_x = max(0, min(img_size[0], point[0] + offset_x))
            new_y = max(0, min(img_size[1], point[1] + offset_y))
            new_points.append((new_x, new_y))
        self.points = tuple(new_points)
