from dataclasses import dataclass
import torch


@dataclass
class HookKeeper:
   name: str
   hook_fn: callable
   hook_handle: None | torch.utils.hooks.RemovableHandle  # None before attached


class LlamaScope:
    """Class for adding, using, and removing PyTorch hooks with a model."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._build_module_dict()


    def _build_module_dict(self):
        """Walks the model's module tree and builds a name: module map."""
        self._module_dict = {}

        def recurse(module, prefix=''):
            """Recursive tree walk to build self._module_dict."""
            for name, child in module.named_children():
                self._module_dict[prefix+name] = child
                recurse(child, prefix=prefix+name+'-')

        recurse(self.model)  # build the tree

    def list_modules(self):
        """Lists all modules in the module dictionary."""
        return self._module_dict.keys()
    
    def add_fwd_hook(self, hook: HookKeeper, module_str: str):
        """Registers hook on the module given by module_str."""
        handle = self._module_dict[module_str].register_forward_hook(hook.hook_fn)
        hook.handle = handle
        self.hooks.append(hook)
