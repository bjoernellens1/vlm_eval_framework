"""Registry system for encoders, heads, and datasets."""

from typing import Any, Callable, Dict, List, Optional, Type

from vlm_eval.core.base_dataset import BaseDataset
from vlm_eval.core.base_encoder import BaseEncoder
from vlm_eval.core.base_dataset import BaseDataset
from vlm_eval.core.base_encoder import BaseEncoder
from vlm_eval.core.base_head import BaseHead, BaseSegmentationHead


class EncoderRegistry:
    """Registry for vision encoders.
    
    Provides a decorator-based plugin system for registering and retrieving
    encoder implementations.
    
    Example:
        >>> @EncoderRegistry.register("my_encoder")
        >>> class MyEncoder(BaseEncoder):
        ...     pass
        ...
        >>> encoder = EncoderRegistry.get("my_encoder", variant="base")
        >>> available = EncoderRegistry.list_available()
    """
    
    _registry: Dict[str, Type[BaseEncoder]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an encoder class.
        
        Args:
            name: Name to register the encoder under.
        
        Returns:
            Decorator function.
        
        Raises:
            ValueError: If name is already registered or class doesn't inherit from BaseEncoder.
        
        Example:
            >>> @EncoderRegistry.register("radio")
            >>> class RadioEncoder(BaseEncoder):
            ...     pass
        """
        def decorator(encoder_cls: Type[BaseEncoder]) -> Type[BaseEncoder]:
            # Type checking
            if not issubclass(encoder_cls, BaseEncoder):
                raise ValueError(
                    f"Encoder class {encoder_cls.__name__} must inherit from BaseEncoder"
                )
            
            # Check for duplicates
            if name in cls._registry:
                raise ValueError(
                    f"Encoder '{name}' is already registered. "
                    f"Registered encoders: {list(cls._registry.keys())}"
                )
            
            cls._registry[name] = encoder_cls
            return encoder_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseEncoder:
        """Get an encoder instance by name.
        
        Args:
            name: Name of the registered encoder.
            **kwargs: Arguments to pass to the encoder constructor.
        
        Returns:
            Encoder instance.
        
        Raises:
            ValueError: If encoder name is not registered.
        
        Example:
            >>> encoder = EncoderRegistry.get("radio", variant="base", pretrained=True)
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Encoder '{name}' not found. Available encoders: {available}"
            )
        
        encoder_cls = cls._registry[name]
        return encoder_cls(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEncoder:
        """Create encoder from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'name' key and other parameters.
        
        Returns:
            Encoder instance.
        
        Example:
            >>> config = {"name": "radio", "variant": "base"}
            >>> encoder = EncoderRegistry.from_config(config)
        """
        config = config.copy()
        name = config.pop("name")
        # Extract and merge kwargs if present
        kwargs = config.pop("kwargs", {})
        config.update(kwargs)
        return cls.get(name, **config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered encoder names.
        
        Returns:
            List of registered encoder names.
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an encoder name is registered.
        
        Args:
            name: Encoder name to check.
        
        Returns:
            True if registered, False otherwise.
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered encoders.
        
        Warning: This is mainly for testing purposes.
        """
        cls._registry.clear()


class HeadRegistry:
    """Registry for segmentation heads.
    
    Provides a decorator-based plugin system for registering and retrieving
    segmentation head implementations.
    
    Example:
        >>> @HeadRegistry.register("linear_probe")
        >>> class LinearProbeHead(BaseSegmentationHead):
        ...     pass
        ...
        >>> head = HeadRegistry.get("linear_probe", encoder=encoder, num_classes=21)
        >>> available = HeadRegistry.list_available()
    """
    
    _registry: Dict[str, Type[BaseHead]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a head class.
        
        Args:
            name: Name to register the head under.
        
        Returns:
            Decorator function.
        
        Raises:
            ValueError: If name is already registered or class doesn't inherit from BaseHead.
        
        Example:
            >>> @HeadRegistry.register("linear_probe")
            >>> class LinearProbeHead(BaseSegmentationHead):
            ...     pass
        """
        def decorator(head_cls: Type[BaseHead]) -> Type[BaseHead]:
            # Type checking
            if not issubclass(head_cls, BaseHead):
                raise ValueError(
                    f"Head class {head_cls.__name__} must inherit from BaseHead"
                )
            
            # Check for duplicates
            if name in cls._registry:
                raise ValueError(
                    f"Head '{name}' is already registered. "
                    f"Registered heads: {list(cls._registry.keys())}"
                )
            
            cls._registry[name] = head_cls
            return head_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str, encoder: BaseEncoder, **kwargs: Any) -> BaseSegmentationHead:
        """Get a head instance by name.
        
        Args:
            name: Name of the registered head.
            encoder: Encoder instance to use.
            **kwargs: Additional arguments to pass to the head constructor.
        
        Returns:
            Head instance.
        
        Raises:
            ValueError: If head name is not registered.
        
        Example:
            >>> head = HeadRegistry.get("linear_probe", encoder=encoder, num_classes=21)
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Head '{name}' not found. Available heads: {available}"
            )
        
        head_cls = cls._registry[name]
        return head_cls(encoder=encoder, **kwargs)
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        encoder: BaseEncoder
    ) -> BaseSegmentationHead:
        """Create head from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'name' key and other parameters.
            encoder: Encoder instance to use.
        
        Returns:
            Head instance.
        
        Example:
            >>> config = {"name": "linear_probe", "num_classes": 21}
            >>> head = HeadRegistry.from_config(config, encoder)
        """
        config = config.copy()
        name = config.pop("name")
        # Extract and merge kwargs if present
        kwargs = config.pop("kwargs", {})
        config.update(kwargs)
        return cls.get(name, encoder=encoder, **config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered head names.
        
        Returns:
            List of registered head names.
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a head name is registered.
        
        Args:
            name: Head name to check.
        
        Returns:
            True if registered, False otherwise.
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered heads.
        
        Warning: This is mainly for testing purposes.
        """
        cls._registry.clear()


class DatasetRegistry:
    """Registry for segmentation datasets.
    
    Provides a decorator-based plugin system for registering and retrieving
    dataset implementations.
    
    Example:
        >>> @DatasetRegistry.register("pascal_voc")
        >>> class PascalVOCDataset(BaseDataset):
        ...     pass
        ...
        >>> dataset = DatasetRegistry.get("pascal_voc", root="/data", split="val")
        >>> available = DatasetRegistry.list_available()
    """
    
    _registry: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a dataset class.
        
        Args:
            name: Name to register the dataset under.
        
        Returns:
            Decorator function.
        
        Raises:
            ValueError: If name is already registered or class doesn't inherit from BaseDataset.
        
        Example:
            >>> @DatasetRegistry.register("pascal_voc")
            >>> class PascalVOCDataset(BaseDataset):
            ...     pass
        """
        def decorator(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
            # Type checking
            if not issubclass(dataset_cls, BaseDataset):
                raise ValueError(
                    f"Dataset class {dataset_cls.__name__} must inherit from BaseDataset"
                )
            
            # Check for duplicates
            if name in cls._registry:
                raise ValueError(
                    f"Dataset '{name}' is already registered. "
                    f"Registered datasets: {list(cls._registry.keys())}"
                )
            
            cls._registry[name] = dataset_cls
            return dataset_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseDataset:
        """Get a dataset instance by name.
        
        Args:
            name: Name of the registered dataset.
            **kwargs: Arguments to pass to the dataset constructor.
        
        Returns:
            Dataset instance.
        
        Raises:
            ValueError: If dataset name is not registered.
        
        Example:
            >>> dataset = DatasetRegistry.get("pascal_voc", root="/data", split="val")
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        
        dataset_cls = cls._registry[name]
        return dataset_cls(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseDataset:
        """Create dataset from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'name' key and other parameters.
        
        Returns:
            Dataset instance.
        
        Example:
            >>> config = {"name": "pascal_voc", "root": "/data", "split": "val"}
            >>> dataset = DatasetRegistry.from_config(config)
        """
        config = config.copy()
        name = config.pop("name")
        # Extract and merge kwargs if present
        kwargs = config.pop("kwargs", {})
        config.update(kwargs)
        return cls.get(name, **config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered dataset names.
        
        Returns:
            List of registered dataset names.
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset name is registered.
        
        Args:
            name: Dataset name to check.
        
        Returns:
            True if registered, False otherwise.
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered datasets.
        
        Warning: This is mainly for testing purposes.
        """
        cls._registry.clear()
