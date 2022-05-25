# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         wd_schedulers.py
# Description:  Weight decay scheduling class
# ===========================================================================
import warnings
import weakref
from functools import wraps

from torch.optim.optimizer import Optimizer

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class _WDScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base weight decays
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_wd', group['weight_decay'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_wd' not in group:
                    raise KeyError("param 'initial_wd' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_wds = [group['initial_wd'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `wd_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has aldeady been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_wd(self):
        """ Return last computed weight decay by current scheduler.
        """
        return self._last_wd

    def get_wd(self):
        # Compute weight decay using chainable form of the scheduler
        raise NotImplementedError

    def print_wd(self, is_verbose, group, wd, epoch=None):
        """Display the current weight decay.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting weight decay'
                      ' of group {} to {:.4e}.'.format(group, wd))
            else:
                print('Epoch {:5d}: adjusting weight decay'
                      ' of group {} to {:.4e}.'.format(epoch, group, wd))

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after weight decay scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`wd_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first wd_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `wd_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `wd_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the weight decay schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_wd_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_wd_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_wd_called_within_step = False

        with _enable_get_wd_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_wd()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_wd"):
                    values = self._get_closed_form_wd()
                else:
                    values = self.get_wd()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, wd = data
            param_group['weight_decay'] = wd
            self.print_wd(self.verbose, i, wd, epoch)

        self._last_wd = [group['weight_decay'] for group in self.optimizer.param_groups]


class StepWD(_WDScheduler):

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size or 1  # If zero, this defaults to 1
        self.gamma = gamma
        super(StepWD, self).__init__(optimizer, last_epoch, verbose)

    def get_wd(self):
        if not self._get_wd_called_within_step:
            warnings.warn("To get the last weight decay computed by the scheduler, "
                          "please use `get_last_wd()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['weight_decay'] for group in self.optimizer.param_groups]
        return [group['weight_decay'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_wd(self):
        return [base_wd * self.gamma ** (self.last_epoch // self.step_size)
                for base_wd in self.base_wds]
