import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super(MyAlexNet, self).__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    # freezing the layers by setting requires_grad=False
    # example: self.cnn_layers[idx].weight.requires_grad = False

    # take care to turn off gradients for both weight and bias

    # raise NotImplementedError('__init__ not implemented')
    model = alexnet(pretrained=True)
    self.cnn_layers = list(model.children())[0]
    i = 0
    for layer in self.cnn_layers:
      if i in [0, 3, 6, 8, 10]:
        layer.weight.requires_grad = False
      i += 1

    self.fc_layers = nn.Sequential(*[model.classifier[i] for i in range(6)],
                                   nn.Linear(in_features=4096, out_features=15, bias=True))

    temp = 0
    for layer in self.fc_layers:
      if temp in [1, 4]:
        layer.weight.grad = None
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
      temp += 1

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
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################

    # raise NotImplementedError('forward not implemented')
    y = self.cnn_layers(x)
    y = y.reshape((y.shape[0], 9216))
    model_output = self.fc_layers(y)
    print(model_output.shape)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
