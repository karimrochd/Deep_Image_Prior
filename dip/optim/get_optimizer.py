import torch

def get_optimizer(params, name='adam', lr=1e-2, weight_decay=0.0, momentum=0.9, nesterov=False):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'lbfgs':
        return torch.optim.LBFGS(params, lr=lr)
    raise ValueError(f'Unknown optimizer: {name}')
