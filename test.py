import rasterio
import numpy as np
from PIL import Image
import tifffile

images = tifffile.imread('/Users/patryk/masters/methane_segmentation/datasets/ang20191025t193513_r12800_c0_w512_h512/mag1c.tif')

print(images)