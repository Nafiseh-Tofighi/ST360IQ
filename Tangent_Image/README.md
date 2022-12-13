
# Tangent images

Parameters of the `configuration.py` file should be changed corresponding to the dataset details. Saliency maps of images in the dataset should be available if the sampling method is based on saliency scores(not randomly).

**Parameters:**
- `source_path`: str.  
The direction to the folder contains images.
- `target_path`: str.  
A direction for writing output Tangent images.
- `smap_path`: str.  
Path of saliency maps. Each image in the dataset must have a pair saliency map with the same name.
- `txt_file_name`: str.  
Direction to the text file containing the list of images in the database and their IQA scores. 
- `n_tan_patch`: int.  
Total number of Tangent images extracted from a single image.
- `sampling_method`: str.  
[top, stochastic, random] ---> selecting the Tangent patches method based on their saliency scores.
  - top: selecting the top number of Tangent images
  - stochastic: Stochasticly chose Tangant patches
  - random: select Tangent images randomly, regardless of their saliency scores
- `sampling_region_size`: int.  
Dimension of the MLP (FeedForward) layer. 
- `tan_stride`: int.
Size of regions in ERP image that their saliency scores are calculated and the tangent images will generate from centers of these regions.
- `fov`: (int, int).  
Field of view for extracting Tangent images.

**In the end, the target folder will contain two sets of files:**

 1. `.pt` files: For each extracted Tangent image, a `.pt` file will be generated. This file includes data on the Tangent image, its center, quality score, and source ERP image.
 2. `tan_imgs_list.txt` file: Including a list of all generated `.pt` files.
