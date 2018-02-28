import torch
from torch.autograd import Variable
import copy


class ClassificationSolver(object):
    """
    The main code structure is like assignment1/cs231n/solver.py
    """
    def __init__(self, loader_train, loader_val, model, loss_fn, optimizer, scheduler=None, dtype=torch.FloatTensor,
                 **kwargs):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dtype = dtype

        self.num_epochs = kwargs.pop('num_epochs', 1)
        self.print_every = kwargs.pop('print_every', 100)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self._reset()


    def _reset(self):
        """Not well implemented yet!"""
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_model_wts = None
        self.best_vac_acc = 0

        # TODO: others reset, e.g. reset model, optimizer etc?!


    def _step(self, x, y):
        """Make a single gradient update given a training batch."""
        self.model.train()

        x_var = Variable(x.type(self.dtype), requires_grad=False)
        y_var = Variable(y.type(self.dtype), requires_grad=False)

        # forward
        scores = self.model(x_var)
        loss = self.loss_fn(scores, y_var)

        # zero the parameter gradients
        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record loss
        self.loss_history.append(loss)


    def _save_checkpoint(self):
        pass


    def check_accuracy(self, loader):
        self.model.eval()

        num_correct = 0
        num_samples = 0

        for x, y in loader:
            x_var, y_var = Variable(x.type(self.dtype), volatile=True), Variable(y.type(self.dtype), volatile=True)

            scores = self.model(x_var)
            _, preds = torch.max(scores, 1)
            num_correct += torch.sum(preds == y.data)
            num_samples += preds.data.size(0)

        return float(num_correct) / num_samples


    def _epoch(self):
        # train one epoch
        if self.scheduler is not None:
            self.scheduler.step()

        for t, (x, y) in enumerate(self.loader_train):
            self._step(x, y)

            # print loss
            if self.verbose and t % self.print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, self.loss_history[-1]))


    def train(self):
        for epoch in range(self.num_epochs):
            self._epoch()

            train_acc = self.check_accuracy(self.loader_train)
            val_acc = self.check_accuracy(self.loader_val)

            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self._save_checkpoint()

            if self.verbose:
                print('(Epoch %d / %d) train acc: %f; val acc: %f' % (epoch, self.num_epochs, train_acc, val_acc))

            # keep track of the best model
            if val_acc > self.best_vac_acc:
                self.best_vac_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # At the end of training swap the best params into the model
        self.model.load_state_dict(self.best_model_wts)







