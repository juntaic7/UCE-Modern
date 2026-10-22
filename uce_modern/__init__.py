from transformers import AutoConfig, AutoModel

from .configuration_uce import UCEConfig
from .modeling_uce import UCEModel, UCEModelOutput, UCEPreTrainedModel

AutoConfig.register("uce", UCEConfig)
AutoModel.register(UCEConfig, UCEModel)

__all__ = ["UCEConfig", "UCEModel", "UCEModelOutput", "UCEPreTrainedModel", "UCEProcessor"]


def __getattr__(name):
    # Lazy import: the processor pulls in scanpy, which the model itself doesn't need.
    if name == "UCEProcessor":
        from .processing_uce import UCEProcessor

        return UCEProcessor
    raise AttributeError(name)
