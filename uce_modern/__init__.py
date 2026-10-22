from transformers import AutoConfig, AutoModel

from .configuration_uce import UCEConfig
from .modeling_uce import UCEModel, UCEModelOutput, UCEPreTrainedModel

AutoConfig.register("uce", UCEConfig)
AutoModel.register(UCEConfig, UCEModel)

__all__ = ["UCEConfig", "UCEModel", "UCEModelOutput", "UCEPreTrainedModel"]
