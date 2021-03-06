import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super(SimpleNetDropout, self).__init__()

    # self.cnn_layers = nn.Sequential()
    # self.fc_layers = nn.Sequential()
    # self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    #raise NotImplementedError('__init__ not implemented')

    self.cnn_layers = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
                                    nn.MaxPool2d(3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
                                    nn.MaxPool2d(3),
                                    nn.ReLU(),
                                    nn.Flatten())

    self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Dropout(), nn.Linear(100, 15))
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    model_output = self.fc_layers(self.cnn_layers(x))

    #raise NotImplementedError('forward not implemented')

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
