import torch

def SGD(model, lr):
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=0)


def Adam(model, lr):
    return torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)