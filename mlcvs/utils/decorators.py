import inspect
import functools
import pytorch_lightning as pl

allowed_hooks = ("on_predict_batch_end","on_predict_batch_start","on_predict_end","on_predict_epoch_end","on_predict_epoch_start","on_predict_start","on_test_batch_end","on_test_batch_start","on_test_end","on_test_epoch_end","on_test_epoch_start","on_test_start","on_train_batch_end","on_train_batch_start","on_train_end","on_train_epoch_end","on_train_epoch_start","on_train_start","on_validation_batch_end","on_validation_batch_start","on_validation_end","on_validation_epoch_end","on_validation_epoch_start","on_validation_start")

"""Iterate through a list of methods in a class and apply decorator."""
def decorate_methods(decorator, methods=allowed_hooks):
    def decorate(cls):
        for name, fn in inspect.getmembers(cls, inspect.isroutine):
            if name in methods :
                setattr(cls, name, decorator(fn))
        return cls

    return decorate

"""Recursively apply hooks based on self.hooks dictionary, e.g. hooks = {"on_predict_batch_end": function_to_be_called]}"""
def apply_hooks(f):
    @functools.wraps(f)
    def wrapper(self,*args, **kwargs):
        # loop over all saved hooks in class
        for hook,funcs in self.hooks.items():
            # check if the current function name matches a saved hook
            # in case execute all the associated funcs
            if hook == f.__name__:
                # if it is a string convert to iterable
                if isinstance(funcs, str):
                    funcs = [ funcs ] 
                for func in funcs:
                    getattr(self,func)()
        return f(self,*args, **kwargs)

    return wrapper

"""Call children hooks """
def call_submodules_hooks(f, allowed_hooks = allowed_hooks):
    @functools.wraps(f)
    def wrapper(self,*args, **kwargs):
        f_name = f.__name__
        if f_name in allowed_hooks:
            for child in self.children():
                if hasattr(child,f_name):
                    func = getattr(child,f_name)
                    func(*args, **kwargs)
        return f(self,*args, **kwargs)

    return wrapper

