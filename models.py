import os
from typing import Optional
from dataclasses import dataclass, field
from union import Resources


@dataclass
class LLMRuntime:
    resources: Resources
    accelerator: Optional[str]
    stream_model: bool
    app_kwargs: dict = field(default_factory=dict)


@dataclass
class CacheRuntime:
    resources: Resources = field(default_factory=lambda: Resources(cpu="3", mem="4Gi"))
    accelerator: Optional[str] = None


@dataclass
class Model:
    display_name: str
    model_id: str
    app_name: Optional[str] = None
    model_uri: Optional[str] = None
    base_url: Optional[str] = None
    base_url_env_var: Optional[str] = None
    llm_runtime: Optional[LLMRuntime] = None
    cache_runtime: Optional[CacheRuntime] = None

    @property
    def endpoint(self) -> str:
        if self.base_url is not None:
            return self.base_url
        elif self.base_url_env_var is not None:
            endpoint = os.getenv(self.base_url_env_var)
            if endpoint is None:
                msg = "base_url_end_var is not set"
                raise RuntimeError(msg)
            return endpoint
        else:
            msg = "base_url or base_url_env must be set"
            raise RuntimeError(msg)


@dataclass
class CacheWorkflow:
    secret_key: str


@dataclass
class LLMConfig:
    models: list[Model]
    cache_workflow: Optional[CacheWorkflow] = None


def get_config() -> LLMConfig:
    from mashumaro.codecs.yaml import YAMLDecoder

    config_file = os.getenv("CONFIG_FILE")
    if config_file is None:
        msg = "CONFIG_FILE must be set"
        raise RuntimeError(msg)

    decoder = YAMLDecoder(LLMConfig)

    with open(config_file) as f:
        config = decoder.decode(f.read())
    return config
