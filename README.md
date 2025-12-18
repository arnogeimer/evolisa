The code for this project is a heavily modified version of https://github.com/itsmux/evolisa.
Added functionalities are round-decreasing offsets when changing polygons, additional mutation possibilities, freezing polygons when improvement are rare coupled with the addition of a new set of polygons (copies of the frozen polygons), andl as improvements to the fitness function and image drawing.

In the image below, the top rows show individual reconstructions based on incomplete information. The bottom row demonstrates how combining outputs yields a more accurate final image.

![Example 1: Combining outputs](img_out/grid_ml.png)

![Example 2: Combining models during training](img_out/grid_gpe.png)
