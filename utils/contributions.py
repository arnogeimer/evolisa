from utils.dna import DNA
from utils.utils import image_to_numpy, fitness
import random

def determine_n_best_polygons_by_fitness_contribution(dna: DNA = None, n: int = 50):
    fitness_contributions = []
    for i in range(len(dna.trainable_polygons)):
        polygon_collection = DNA(
            img_path=dna.original_img_path,
            trainable_polygons=dna.polygons[:i] + self.dna.polygons[i + 1 :],
        )
        img_polygon = polygon_collection.draw(save=False)
        fitness_polygon = fitness(self.image, img_polygon)
        fitness_contributions.append(self.fitness - fitness_polygon)
    del img_polygon
    zipped = sorted(
        zip(fitness_contributions, self.dna.polygons),
        reverse=True,
        key=lambda x: x[0],
    )
    best_polygons = [item[1] for item in zipped[:n]]
    return best_polygons


def determine_n_polygons_by_fitness(img_size, original_image, n: int = 50, polygons: List[Polygon] = None, reverse: bool = False):
    fitness_contributions = []
    for polygon in polygons:
        polygon_collection = DNA(
            img_size=img_size,
            img_path=None,
            trainable_polygons=[polygon],
        )
        original_numpy = image_to_numpy(original_image)
        fitness_polygon = fitness(original_numpy, image_to_numpy(polygon_collection.draw(save=False, draw_full=False))).item()
        fitness_contributions.append(fitness_polygon)
    zipped = sorted(
        zip(fitness_contributions, polygons),
        reverse=reverse,
        key=lambda x: x[0],
    )
    best_polygons = [item[1] for item in zipped[:n]]
    return best_polygons


def determine_n_best_polygons_randomly(self, n: int = 50):
    return random.sample(self.dna.polygons, n)