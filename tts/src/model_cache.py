"""Model cache management for VoxCPM."""

import asyncio
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from voxcpm import VoxCPM
    from .config import ModelConfig


def _patch_torch_load():
    """
    Patch torch.load to set weights_only=False for compatibility.

    PyTorch 2.6+ changed the default value of weights_only from False to True,
    which breaks loading of legacy .tar format models used by VoxCPM.
    """
    import torch

    # Get the original torch.load function
    original_load = torch.load

    # Define a wrapper that always sets weights_only=False
    def patched_load(f, *args, **kwargs):
        # Force weights_only=False for compatibility
        kwargs['weights_only'] = False
        return original_load(f, *args, **kwargs)

    # Replace torch.load with our patched version
    torch.load = patched_load


def _disable_torch_compile():
    """
    Disable torch.compile to avoid TLS issues in async executor.

    PyTorch's torch.compile uses thread-local storage which doesn't work
    well when the model is loaded in an executor and used in the main thread.
    """
    import torch
    import torch._dynamo as dynamo

    # Disable torch.compile globally
    dynamo.config.cache_size_limit = 0


# Apply patches once when the module is imported
_patch_torch_load()
_disable_torch_compile()


class ModelCache:
    """
    Singleton cache for VoxCPM model instances.

    Manages loading and caching of VoxCPM models to avoid
    reloading the model for each request.
    """

    _instance: Optional["ModelCache"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        """Initialize the model cache."""
        self._models: dict[str, "VoxCPM"] = {}
        self._loading: dict[str, asyncio.Lock] = {}

    @classmethod
    async def get_instance(cls) -> "ModelCache":
        """
        Get the singleton ModelCache instance.

        Returns:
            ModelCache instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def get_model(self, config: "ModelConfig") -> "VoxCPM":
        """
        Get or load a VoxCPM model.

        Args:
            config: Model configuration

        Returns:
            VoxCPM model instance
        """
        model_key = config.model_name

        # Check if already loaded
        if model_key in self._models:
            return self._models[model_key]

        # Check if currently being loaded
        if model_key in self._loading:
            await self._loading[model_key].acquire()
            self._loading[model_key].release()
            return self._models[model_key]

        # Start loading
        self._loading[model_key] = asyncio.Lock()
        await self._loading[model_key].acquire()

        try:
            # Load model in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                self._load_model,
                config
            )

            self._models[model_key] = model
            return model
        finally:
            self._loading[model_key].release()
            if model_key in self._loading and not self._loading[model_key].locked():
                del self._loading[model_key]

    def _load_model(self, config: "ModelConfig") -> "VoxCPM":
        """
        Load VoxCPM model (runs in executor).

        Args:
            config: Model configuration

        Returns:
            Loaded VoxCPM model instance
        """
        import os
        import torch
        from voxcpm import VoxCPM

        # Use the model path directly
        model_path = config.model_name

        # If it's a relative path, resolve it properly
        if not os.path.isabs(model_path):
            # Try current directory first
            if os.path.exists(model_path):
                model_path = os.path.abspath(model_path)
            else:
                # Try project root (parent of src directory)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                potential_path = os.path.join(project_root, model_path)
                if os.path.exists(potential_path):
                    model_path = potential_path
                else:
                    # If local path doesn't exist, it might be a HuggingFace model ID
                    # Keep as-is and let VoxCPM.from_pretrained handle it
                    pass

        # Set device for torch before loading the model
        if config.device and config.device == "cuda" and torch.cuda.is_available():
            # VoxCPM will automatically use CUDA when available
            pass

        # For local models, VoxCPM.from_pretrained works with local paths
        model = VoxCPM.from_pretrained(model_path)

        return model

    async def reload_model(self, config: "ModelConfig") -> "VoxCPM":
        """
        Force reload a model.

        Args:
            config: Model configuration

        Returns:
            Newly loaded VoxCPM model instance
        """
        model_key = config.model_name

        # Remove from cache
        if model_key in self._models:
            del self._models[model_key]

        return await self.get_model(config)

    def is_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_name: Model name

        Returns:
            True if model is loaded
        """
        return model_name in self._models

    @property
    def loaded_models(self) -> list[str]:
        """Get list of loaded model names."""
        return list(self._models.keys())
