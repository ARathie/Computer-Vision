import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################

  img_dir = dir_name + "*/*/*.jpg"
  scale = StandardScaler()
  for file in glob.glob(img_dir):
    img = Image.open(file).convert('L')
    img = np.array(img)
    img = img.astype(np.float32) / 255
    img = img.flatten()
    img = img.reshape(-1, 1)
    scale.partial_fit(img)
    StandardScaler(copy=True, with_mean=True, with_std=True)
  mean = scale.mean_
  std = scale.scale_

  #raise NotImplementedError('compute_mean_and_std not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
