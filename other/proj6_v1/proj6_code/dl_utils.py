'''
Utilities to be used along with the deep model
'''

import torch
import torch.nn as nn

def predict_labels(model: torch.nn.Module, x: torch.tensor) -> torch.tensor:
  '''
  Perform the forward pass and extract the labels from the model output

  Args:
  -   model: a model (which inherits from nn.Module)
  -   x: the input image [Dim: (N,C,H,W)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''

  predicted_labels = None

  #############################################################################
  # Student code begin
  #############################################################################

  # raise NotImplementedError('predict_labels not implemented')
  # output = model.forward(x)
  # labels = torch.argmax(output, 1)
  
  # N = x.shape[0]
  # labels = torch.zeros(N)
  # i = 0
  # for img in x:
  #   output = model.forward(img.unsqueeze(0))
  #   values, indices = torch.max(output, 1)
  #   labels[i] = int(indices[0].item())
  #   i += 1

  new_model = model(x)
  predicted_labels = []
  for m in range(new_model.shape[0]):
    predicted_labels.append(torch.argmax(new_model[m]))
  predicted_labels = torch.tensor(predicted_labels)

  #print('predicted_labels: ', labels)
  # predicted_labels = labels
  #############################################################################
  # Student code end
  #############################################################################
  return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Args:
  -   model: a model (which inherits from nn.Module)
  -   model_output: the raw scores output by the net
  -   target_labels: the ground truth class labels
  -   is_normalize: bool flag indicating that loss should be divided by the batch size
  Returns:
  -   the loss value
  '''

  loss = None

  #############################################################################
  # Student code begin
  #############################################################################

  # raise NotImplementedError('compute_loss not implemented')
  # batch_size, _ = model_output.shape
  # loss_fn = nn.CrossEntropyLoss()
  # loss = loss_fn(model_output, target_labels)
  # #loss = model.loss_criterion(model_output, target_labels)

  # if is_normalize:
  #   loss /= batch_size

  entropy_loss = torch.nn.CrossEntropyLoss()
  loss = entropy_loss(model_output, target_labels)
  if is_normalize:
    loss = loss / model_output.shape[0]
  #############################################################################
  # Student code end
  #############################################################################
  return loss
