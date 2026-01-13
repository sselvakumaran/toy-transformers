"""Model registry and auto-loading.

Import this module to register all available models:
    from toy_transformers.models import ModelRegistry

    # List available models
    print(ModelRegistry.list_models())

    # Describe a model
    print(ModelRegistry.describe('gpt-v2'))

    # Build a model
    model = ModelRegistry.build('gpt-v2', config_dict, data_config)
"""

# Import all model modules to trigger registration
from toy_transformers.models import gptv1, gptv2, gptv3
from toy_transformers.models.registry import ModelRegistry

__all__ = ['ModelRegistry', 'gptv1', 'gptv2', 'gptv3']
