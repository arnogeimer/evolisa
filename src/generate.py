import copy
import glob
import os
import random

import cupy as cp
import numpy as np
import tqdm
from PIL import Image

from utils.dna import DNA, generate_dna
from utils.utils import (
    ParameterManager,
    combine_images,
    generate_gif_from_output_images,
    image_to_numpy,
    load_image,
    oit_composite,
)

def fitness(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    fitness function determines how much alike an image (as ndarray) and DNA are.
    """
    return cp.linalg.norm(cp.array(img_1) - cp.array(img_2))


class ImageGenerator:
    def __init__(
        self,
        im_path: str,
        name: str = "nameless",
        draw_on_init: bool = True,
        num_polygons: int = 50,
    ):
        self.name: str = name
        self.im_path: str = im_path
        self.image: Image.Image = load_image(im_path)
        self.img_size: tuple = self.image.size
        self.dna: DNA = generate_dna(
            self.img_size, self.im_path, dna_size=num_polygons, fixed_colour=False
        )
        self.frozen_polygons: tuple[int, int] = (0, 0)
        if draw_on_init:
            self.dna.draw(save=True, name=f"{self.name}_initial")

    def evolve(
        self,
        num_epochs: int,
        round: int,
        num_mutations: int = 1,
        offsetcol: int = 10,
        offsetx: int = 10,
        offsety: int = 10,
    ):
        original_numpy = image_to_numpy(self.image)
        num_changes = 0
        for _ in tqdm.trange(num_epochs):
            dna_child = self.dna.mutate_dna(
                num_mutations=num_mutations,
                offsetcol=offsetcol,
                offsetx=offsetx,
                offsety=offsety,
                frozen_polygons=self.frozen_polygons,
            )
            fitness_parent = fitness(
                original_numpy, image_to_numpy(self.dna.draw(save=False))
            )
            fitness_child = fitness(
                original_numpy, image_to_numpy(dna_child.draw(save=False))
            )
            if fitness_child < fitness_parent:
                self.dna = dna_child
                fitness_parent = fitness_child
                num_changes += 1
        self.fitness = fitness_parent
        print(f"{self.name} made {num_changes} improvements in round {round}.")
        return num_changes

    def save_drawn_image(self, name: str):
        self.dna.draw(save=True, name=f"{name}")

    '''def determine_n_best_polygons_by_fitness_contribution(self, n: int = 50):
        fitness_contributions = []
        for i in range(len(self.dna.polygons)):
            polygon_collection = DNA(
                img_size=self.img_size,
                img_path=self.im_path,
                polygons=self.dna.polygons[:i] + self.dna.polygons[i + 1 :],
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


    def determine_n_best_polygons_by_fitness(self, n: int = 50, dna: DNA = None):
        fitness_contributions = []
        for polygon in dna.polygons:
            polygon_collection = DNA(
                img_size=self.img_size,
                img_path=self.im_path,
                polygons=[polygon],
            )
            original_numpy = image_to_numpy(self.image)
            fitness_polygon = fitness(original_numpy, image_to_numpy(polygon_collection.draw(save=False)))
            fitness_contributions.append(original_numpy - fitness_polygon)
        del img_polygon
        zipped = sorted(
            zip(fitness_contributions, self.dna.polygons),
            reverse=False,
            key=lambda x: x[0],
        )
        best_polygons = [item[1] for item in zipped[:n]]
        return best_polygons
    
    def determine_n_best_polygons_randomly(self, n: int = 50):
        return random.sample(self.dna.polygons, n)'''


def cleanup_temp_files():
    files = glob.glob("./img_out/*.png")
    for f in files:
        os.remove(f)


ROUNDS = 200


def add_dna(client: ImageGenerator, num_polygons: int = 50) -> ImageGenerator:
    """Copy some polygons and halve their and their parents' alpha to add more DNA to the client."""
    duplicate_indices = sorted(
        random.sample(range(len(client.dna.polygons)), num_polygons)
    )
    new_polygons = [copy.deepcopy(client.dna.polygons[i]) for i in duplicate_indices]
    for polygon in new_polygons:
        polygon.halve_alpha()
    for i in duplicate_indices:
        client.dna.polygons[i].halve_alpha()
    client.dna.polygons += new_polygons
    client.frozen_polygons = (0, int(len(client.dna.polygons) - num_polygons))
    return client


def evolisa(
    client: ImageGenerator = None,
    num_epochs: int = 200,
    num_rounds: int = ROUNDS,
    num_polygons: int = 50,
    name: str = "evolisa",
    path: str = "img_in/images.jpg",
):
    if client is None:
        client = ImageGenerator(
            path, name=name, draw_on_init=True, num_polygons=num_polygons
        )
    param_manager = ParameterManager(initial_offset=100)
    for round in range(1, num_rounds + 1):
        offset = param_manager.get_offset()
        num_changes = client.evolve(
            num_epochs=num_epochs,
            round=round,
            offsetcol=offset,
            offsetx=offset,
            offsety=offset,
        )
        print(f"Client fitness after round {round}: {client.fitness}, offset: {offset}")
        if param_manager.should_add_dna(num_changes=num_changes, threshold=3):
            print(
                f"Adding more DNA to client {client.name} at round {round} due to low improvements."
            )
            # client.save_drawn_image(name=f"dna_{name}_round_{round}_before_adding_dna")
            client = add_dna(client, num_polygons=num_polygons)
            # client.save_drawn_image(name=f"dna_{name}_round_{round}_after_adding_dna")
            print(f"Total polygons after adding: {len(client.dna.polygons)}")
        # client.save_drawn_image(name=f"dna_{name}_round_{round}")
    # generate_gif_from_output_images(name=name, num_rounds=num_rounds)
    return client

# We generate a polygon picture from both mona_lisa1.jpg and mona_lisa2.jpg, and combine them into a more correct drawing.

def determine_n_polygons_by_fitness(img_size, original_image, n: int = 50, dna: DNA = None, reverse: bool = False):
        fitness_contributions = []
        for polygon in dna.polygons:
            polygon_collection = DNA(
                img_size=img_size,
                img_path=None,
                polygons=[polygon],
            )
            original_numpy = image_to_numpy(original_image)
            fitness_polygon = fitness(original_numpy, image_to_numpy(polygon_collection.draw(save=False))).item()
            fitness_contributions.append(fitness_polygon)
        del polygon_collection
        zipped = sorted(
            zip(fitness_contributions, dna.polygons),
            reverse=reverse,
            key=lambda x: x[0],
        )
        best_polygons = [item[1] for item in zipped[:n]]
        return best_polygons

NUM_ROUNDS = 20

client1 = evolisa(
    num_epochs=100,
    num_rounds=NUM_ROUNDS,
    num_polygons=10,
    name="amg_1",
    path="img_in/am_goth1.jpg",
)
client2 = evolisa(
    num_epochs=100,
    num_rounds=NUM_ROUNDS,
    num_polygons=10,
    name="amg_2",
    path="img_in/am_goth2.jpg",
)
for i in range(1, 21):

    print(f"ROUND {i}")

    # Add the worst-performing polygons from the other client to oneself
    client1.dna.polygons += determine_n_polygons_by_fitness(img_size=client1.img_size, original_image = client1.image, n = 5, dna = client2.dna, reverse = True)
    client2.dna.polygons += determine_n_polygons_by_fitness(img_size=client1.img_size, original_image = client2.image, n = 5, dna = client1.dna, reverse = True)
    dna1, dna2 = client1.dna.polygons, client2.dna.polygons

    for dna, save_name in zip(
        [dna1, dna2],
        [
            f"./img_out/amg1_round_{i}.png",
            f"./img_out/amg2_round_{i}.png",
        ],
    ):

        img = oit_composite(dna, client1.img_size)
        img.save(save_name)

    client1 = evolisa(
        client=client1,
        num_epochs=100,
        num_rounds=NUM_ROUNDS,
        num_polygons=10,
        name="ml_1",
        path="img_in/am_goth1.jpg",
    )
    client2 = evolisa(
        client=client2,
        num_epochs=100,
        num_rounds=NUM_ROUNDS,
        num_polygons=10,
        name="ml_2",
        path="img_in/am_goth2.jpg",
    )


client1.dna.polygons += determine_n_polygons_by_fitness(img_size=client1.img_size, original_image = client1.image, n = int(len(client2.dna.polygons)), dna = client2.dna, reverse = True)
client2.dna.polygons += determine_n_polygons_by_fitness(img_size=client1.img_size, original_image = client2.image, n = int(len(client2.dna.polygons)), dna = client1.dna, reverse = True)
dna1, dna2 = client1.dna.polygons, client2.dna.polygons

for dna, save_name in zip(
    [dna1, dna2],
    [
        f"./img_out/amg1_round_{i+1}.png",
        f"./img_out/amg2_round_{i+1}.png",
    ],
):

    img = oit_composite(dna, client1.img_size)
    img.save(save_name)

#combine_images()
#cleanup_temp_files()
