from torch import optim


class Optimizer(object):
    def __init__(self, parameters, args):
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
        else:
            self.optimizer = optim.Adam(parameters, lr=args.lr, momentum=args.momentum)

    def get_optimizer(self):
        return self.optimizer