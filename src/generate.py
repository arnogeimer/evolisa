import glob
import os

import tqdm
import time

from utils.dna import DNA
from utils.utils import (
    ParameterManager,
)

import random


def cleanup_temp_files():
    files = glob.glob("./img_out/*.png")
    for f in files:
        os.remove(f)


ROUNDS = 200


# We generate a polygon picture from both mona_lisa1.jpg and mona_lisa2.jpg, and combine them into a more correct drawing.

NUM_ROUNDS = 5

dna1 = DNA(img_path="img_in/gpe1.jpg")
dna2 = DNA(img_path="img_in/gpe2.jpg")

#print(len(dna1.fixed_polygons), len(dna1.trainable_polygons)), print(len(dna2.fixed_polygons), len(dna2.trainable_polygons))
param_manager1 = ParameterManager(initial_offset=100)
param_manager2 = ParameterManager(initial_offset=100)

starttime = time.time()
dna1.generate_dna(dna_size=20), dna2.generate_dna(dna_size=20)

for round in range(150):
    for dna, param_manager in zip(
        [dna1, dna2],
        [param_manager1, param_manager2]
    ):
        offset = param_manager.get_offset()
        num_changes = dna.evolve(num_epochs=2500, offsetcol = param_manager.get_offset(), offsetx=param_manager.get_offset(), offsety=param_manager.get_offset())
        print(f"dna fitness after round {round}: {dna.fitness}, offset: {offset}, changes: {num_changes}")
        if param_manager.should_copy_polygons(num_changes=num_changes, threshold=3):
            print(
                f"Adding more DNA to client 1 at round {round} due to low improvements."
            )
            dna.copy_polygons(num_polygons=10)
            print(f"Total polygons after adding: {len(dna1.fixed_polygons) + len(dna2.trainable_polygons)}")
    dna1.copy_polygons(2)
    dna2.copy_polygons(2)
    dna1.fixed_polygons = dna1.fixed_polygons + random.sample(dna2.trainable_polygons, 2)
    dna2.fixed_polygons = dna2.fixed_polygons + random.sample(dna1.trainable_polygons, 2)
    print(len(dna1.fixed_polygons), len(dna1.trainable_polygons)), print(len(dna2.fixed_polygons), len(dna2.trainable_polygons))

    img1, img2 = dna1.save_current_draw(f"gpe1_round_{round}"), dna2.save_current_draw(f"gpe2_round_{round}")

print(f"Finished computation, total running time of {round((time.time() - starttime) / 60, 0)} minutes")

#combine_images()
#cleanup_temp_files()

