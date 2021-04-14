import glob
import os
import numpy as np
import math

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

  # raise NotImplementedError('compute_mean_and_std not implemented')

  scaler = StandardScaler()

  for root, dirs, files in os.walk(dir_name):
      for file in files:
        if file.endswith('.jpg'):
          im_pic = Image.open(os.path.join(root, file)).convert('L')
          im = np.array(im_pic)/255.0
          im_ft = im.flatten().reshape(-1, 1)
          scaler.partial_fit(im_ft)
  
  mean = scaler.mean_
  std = [math.sqrt(scaler.var_)]

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
