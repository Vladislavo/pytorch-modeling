import torch
from torch import nn

# Very general outline for model definition using PyTorch.

# models are usually defined using subclasses of nn.Module class
class LogisticRegression(nn.Module):
    # nargs number of inpur variables for the logistic regression
    def __init__(self, nargs) -> None:
        super().__init__()
        # define model architecture
        # sequential means layering on in a sequential manner: 
        #   first linear layer and then 
        #   a sigmoid activation function
        self.log_reg = nn.Sequential (
            nn.Linear(nargs, 1), # nargs is the size of the input going in, 1 output from all these inputs
            nn.Sigmoid()
        )

    # forward method means what your model does on the forward pass given inputs to produce an output
    # what happens when an input is introduced, what output the model will return
    def forward(self, x):
        return self.log_reg(x)

# Training a model in PyTorch
# x - inputs, y - ground truth
def train(x, y, n_epochs):
    # initialize the model (with 16 input vars)
    model = LogisticRegression(16)
    # define a cost function
    criterion = nn.BCELoss()
    # choose the optimizer to use (SGD - Stochastic Gradient Descent)
    # model.parameters is theta values updated during the training
    # lr - learning rate (hyper parameter)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # training loop
    for t in range(n_epochs):
        # Forward Propagation
        # get a prediction from the model given the input (pick up the input)
        y_pred = model(x[0])
        # compare the prediction to the ground truth
        loss = criterion(y_pred, y[0])
        # Back Propagation
        # zero out the gradients from before to ensure it is clean and good to go
        optimizer.zero_grad()
        # prepare the back propagation step
        loss.backward()
        # use SGD to update the parameters with lr learning rate
        optimizer.step()
