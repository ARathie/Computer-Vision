#useful util functions implemented with pytorch

import torch
from torch import nn
import numpy as np
"""
Image gradients are needed for both SIFT and the Harris Corner Detector, so we
implement the necessary code only once, here.
"""


class ImageGradientsLayer(torch.nn.Module):
    """
    ImageGradientsLayer: Compute image gradients Ix & Iy. This can be
    approximated by convolving with Sobel filter.
    """
    def __init__(self):
        super(ImageGradientsLayer, self).__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
            bias=False, padding=(1,1), padding_mode='zeros')

        # Instead of learning weight parameters, here we set the filter to be
        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of ImageGradientsLayer. We'll test with a
        single-channel image, and 1 image at a time (batch size = 1).

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, (num_image, 2, height, width)
            tensor for Ix and Iy, respectively.
        """
        return self.conv2d(x)


def get_gaussian_kernel(ksize=7, sigma=5) -> torch.nn.Parameter:
    """
    Generate a Gaussian kernel to be used in HarrisNet for calculating a second moment matrix
    (SecondMomentMatrixLayer). You can call this function to get the 2D gaussian filter.
    
    Since you already implement this in Proj1 we won't be grading this part again, but it is 
    important that you get the correct value here in order to pass the unit tests for HarrisNet.
    
    This might be useful:
    1) We suggest using the outer product trick, it's faster and simpler. And you are less likely to mess up
    the value. 
    2) Make sure the value sum to 1
    3) Some useful torch functions: 
    - torch.mm https://pytorch.org/docs/stable/torch.html#torch.mm 
    - torch.t https://pytorch.org/docs/stable/torch.html#torch.t
    4) Similar to get_sobel_xy_parameters, you should return the filter in torch.nn.Parameter. 
    

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: torch.nn.Parameter of size [ksize, ksize]
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    gaussian_size_k = ksize
    mean = gaussian_size_k // 2
    std_dev = int(sigma)
    array = np.zeros(gaussian_size_k)

    for x in range(gaussian_size_k):
        array[x] = (1 / (np.sqrt(2 * np.pi) * std_dev)) * (np.exp((-1 / (2 * np.square(std_dev))) * np.square(x - mean)))

    array = array / sum(array)

    kernel = np.outer(array, array)

    kernel = torch.Tensor(kernel)
    kernel = nn.Parameter(kernel)


    ### END OF STUDENT CODE ####
    ############################
    return kernel


def get_sobel_xy_parameters() -> torch.nn.Parameter:
    """
    Populate the conv layer weights for the Sobel layer (image gradient
    approximation).

    There should be two sets of filters: each should have size (1 x 3 x 3)
    for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    the Sobel_x filter first, and the Sobel_y filter second.

    Args:
    -   None
    Returns:
    -   kernel: Torch parameter representing (2, 1, 3, 3) conv filters
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    x_layer = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_layer = np.transpose(x_layer)
    combined_layer = np.array([[x_layer], [y_layer]])
    combined_tensor = torch.as_tensor(combined_layer, dtype=float)
    combined_tensor = combined_tensor.float()
    kernel = nn.Parameter(combined_tensor)

    ### END OF STUDENT CODE ####
    ############################

    return kernel
