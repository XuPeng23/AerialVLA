import torch

class BaseModelWrapper:
    def __init__(self):
        self.model: torch.Module = None
        pass
    
    def prepare_inputs(self, episodes):
        pass
    
    def eval(self):
        pass
    
    def run(self, *args, **kwds):
        pass
    