import importlib
import logging
from enum import Enum
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    POLICY = "policy"
    RUNNER = "runner"

    @classmethod
    def from_str(cls, value: str) -> 'ComponentType':
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown component type: {value}")


class Registry:
    """Registry for dynamically loading components"""

    def __init__(self):
        # Initialize registry containers for different component types
        self._components = {ct: {} for ct in ComponentType}

    def register_component(self, component_type: ComponentType, name: str):
        """Generic decorator to register a component class"""
        def _register(cls):
            self._components[component_type][name] = cls
            return cls
        return _register

    def get_component(self, component_type: ComponentType, name: str):
        """Get component class by type and name"""
        components = self._components[component_type]
        if name not in components:
            raise ValueError(f"Unknown {component_type.value}: {name}")
        return components[name]

    def list_components(self, component_type: ComponentType) -> list[str]:
        """List all registered components of a given type"""
        return list(self._components[component_type].keys())

    # Convenience methods for common component types
    def register_policy(self, name: str):
        """Decorator to register a policy class"""
        return self.register_component(ComponentType.POLICY, name)

    def register_runner(self, name: str):
        """Decorator to register a runner class"""
        return self.register_component(ComponentType.RUNNER, name)

    def get_policy(self, name: str):
        """Get policy class by name"""
        return self.get_component(ComponentType.POLICY, name)

    def get_runner(self, name: str):
        """Get runner class by name"""
        return self.get_component(ComponentType.RUNNER, name)

    def register_from_module(self, component_type: ComponentType, module_path: str, suffix: str = ''):
        """Register components from a module path

        Args:
            component_type: Type of component from ComponentType enum
            module_path: Path to the module (e.g., 'flare.runners')
            suffix: Suffix to strip from class names (e.g., 'runner')
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Could not import module {module_path}: {e}")
            return self

        register_method = getattr(self, f"register_{component_type.value}", None)
        if register_method is None:
            def register_method(name): return self.register_component(component_type, name)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.lower().endswith(suffix.lower()):
                name = attr_name[:-(len(suffix))] if suffix else attr_name
                name = name.lower()
                register_method(name)(attr)
                logger.debug(f"Registered {component_type.value}: {name}")

        return self


# Create a global registry instance
registry = Registry()

# Register built-in components
registry.register_from_module(ComponentType.POLICY, 'flare.policies', 'policy')
registry.register_from_module(ComponentType.RUNNER, 'flare.runners', 'runner')


def create_policy(cfg: DictConfig) -> torch.nn.Module:
    policy_name = cfg.policy.name
    policy_cls = registry.get_policy(policy_name)
    policy = policy_cls(cfg).to(cfg.device)
    return policy


def get_policy_class(name: str):
    return registry.get_policy(name)


def get_runner_class(name: str):
    return registry.get_runner(name)


def create_runner(cfg: DictConfig):
    policy = create_policy(cfg)
    runner_cls = get_runner_class(cfg.policy.runner)
    runner = runner_cls(
        config=cfg,
        policy=policy,
        device=cfg.device
    )
    return runner
