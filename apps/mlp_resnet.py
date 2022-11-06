import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim))
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        modules += [ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob)]
    modules += [nn.Linear(hidden_dim, num_classes)]
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    loss_func = nn.SoftmaxLoss()
    error_rate = 0
    sum_loss = 0
    num_samples = len(dataloader.dataset)
    for i, batch in enumerate(dataloader):
        if model.training:
            opt.reset_grad()
        batch_x, batch_y = batch[0], batch[1]
        out = model(nn.ops.reshape(batch_x, (batch_x.shape[0], 784)))
        loss = loss_func(out, batch_y)
        # if eval mode, we dont update parameters
        if model.training:
            loss.backward()
            opt.step()
        # cal some metrics
        preds = np.argmax(out.numpy(), axis=1)
        gts = batch_y.numpy()
        error_rate += np.sum(preds!=gts)
        sum_loss += loss.data.sum().numpy() * batch_x.shape[0]
    error_rate /= num_samples
    return error_rate, sum_loss/num_samples
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    #train epochs 
    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt=opt)
    train_acc = 1-train_error
    #test epoch
    test_error, test_loss = epoch(test_dataloader, model)
    test_acc = 1-test_error
    return train_acc, train_loss, test_acc, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
