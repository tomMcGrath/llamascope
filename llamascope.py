from dataclasses import dataclass
import torch


class LlamaScope:
    """Class for adding, using, and removing PyTorch hooks with a model."""

    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self.override_store = {}
        self._build_module_dict()

    """Module listing."""
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
    
    """Generic hook registration"""
    def add_hook(self, hook_fn, module_str, hook_name):
        """Add a hook_fn to the module given by module_str."""
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle
    
    """Activations caching"""
    def _build_caching_hook(self, module_str):
        self.activations_cache[module_str] = []
        def hook_fn(model, input, output):
            self.activations_cache[module_str].append(output)

        return hook_fn

    def add_caching_hook(self, module_str):
        """Adds an activations caching hook at the location in module_str."""
        hook_fn = self._build_caching_hook(module_str)
        self.add_hook(hook_fn, module_str, 'cache-'+module_str)

    def clear_cache(self, module_str):
        """Clears the activations cache corresponding to module_str."""
        if module_str not in self.activations_cache.keys():
            raise KeyError(f'No activations cache for {module_str}.')
        
        else:
            self.activations_cache[module_str] = []

    def clear_all_caches(self):
        """Clear all activation caches."""
        for module_str in self.activations_cache.keys():
            self.clear_cache(module_str)

    def remove_cache(self, module_str):
        """Remove the cache for module_str."""
        del self.activations_cache[module_str]

    def remove_all_caches(self):
        """Remove all caches."""
        caches = list(self.activations_cache.keys())
        for cache_str in caches:
            self.remove_cache(cache_str)

    """Activation override"""
    def _build_override_hook(self, module_str):
        self.override_store[module_str] = None  # won't override when returned
        def hook_fn(model, input, output):
            return self.override_store[module_str]
        
        return hook_fn
    
    def add_override_hook(self, module_str):
        """Adds hook to overrides output of module_str using override_store"""
        hook_fn = self._build_override_hook(module_str)
        self.add_hook(hook_fn, module_str, 'override-'+module_str)

    def override(self, module_str, override_tensor):
        """Sets the override tensor for module_str."""
        self.override_store[module_str] = override_tensor

    def clear_override(self, module_str):
        """Clear override hook so it won't affect forward pass."""
        self.override_store[module_str] = None

    def clear_all_overrides(self):
        """Clear all override hooks."""
        overrides = list(self.override_store.keys())
        for override in overrides:
            self.clear_override(override)

    """Hook clearup"""
    def remove_hook(self, hook_name):
        """Remove a hook with name hook_name from the model."""
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_all_hooks(self):
        """Remove all hooks from the model."""
        hooks = list(self.hooks.keys())
        for hook_name in hooks:
            self.remove_hook(hook_name)
