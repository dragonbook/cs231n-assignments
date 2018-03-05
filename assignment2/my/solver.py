import torch
from torch.autograd import Variable
import copy


# TODO:
# (1) track ratio of weights:updates
class ClassificationSolver(object):
    """
    The main code structure is like assignment1/cs231n/solver.py
    """
    def __init__(self, loader_train, model, loss_fn, optimizer, dtype, loader_val=None, scheduler=None, **kwargs):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dtype = dtype

        self.num_train_examples = kwargs.pop('num_train_examples', 1000)
        self.num_val_examples = kwargs.pop('num_val_examples', None)
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
        y_var = Variable(y.type(self.dtype).long(), requires_grad=False)

        # forward
        scores = self.model(x_var)
        loss = self.loss_fn(scores, y_var)

        # zero the parameter gradients
        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record loss
        self.loss_history.append(loss.data.cpu())


    def _save_checkpoint(self):
        pass


    def check_accuracy(self, loader, num_examples=None):
        self.model.eval()

        num_correct = 0
        num_samples = 0

        for x, y in loader:
            x_var = Variable(x.type(self.dtype), volatile=True)
            y_var = Variable(y.type(self.dtype).long(), volatile=True)

            scores = self.model(x_var)
            _, preds = torch.max(scores, 1)
            num_correct += torch.sum(preds.data == y_var.data)
            num_samples += preds.data.size(0)

            if num_examples is not None and num_samples >= num_examples:
                break

        return float(num_correct) / num_samples


    def _epoch(self):
        # train one epoch
        if self.scheduler is not None:
            self.scheduler.step()

        for t, (x, y) in enumerate(self.loader_train):
            self._step(x, y)

            # print loss
            if self.verbose and t % self.print_every == 0:
                print('t = %d, loss = %.4f' % (t, self.loss_history[-1]))


    def train(self):
        for epoch in range(self.num_epochs):
            self._epoch()

            train_acc = self.check_accuracy(self.loader_train, self.num_train_examples)
            val_acc = self.check_accuracy(self.loader_val, self.num_val_examples)

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







