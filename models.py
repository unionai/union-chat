import os
from typing import Optional
from dataclasses import dataclass, field
from union import Resources
from flytekit.extras.accelerators import GPUAccelerator


@dataclass
class LLMRuntime:
    resources: Resources
    accelerator: Optional[str]
    stream_model: bool
    llm_type: str
    extra_args: str = ""
    app_kwargs: dict = field(default_factory=dict)


@dataclass
class Model:
    display_name: str
    model_id: str
    name: Optional[str] = None
    model_uri: Optional[str] = None
    base_url: Optional[str] = None
    base_url_env_var: Optional[str] = None
    llm_runtime: Optional[LLMRuntime] = None
    cache_version: str = "v1"

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
            msg = "base_url or base_url_env_var must be set"
            raise RuntimeError(msg)


@dataclass
class CacheWorkflow:
    hf_secret_key: str
    union_secret_key: str
    chunk_size: int = 8 * 1024 * 1024
    resources: Resources = field(default_factory=lambda: Resources(cpu="3", mem="4Gi"))
    accelerator: Optional[str] = None

    @property
    def accelerator_obj(self) -> Optional[GPUAccelerator]:
        if self.accelerator is None:
            return None
        return GPUAccelerator(self.accelerator)


@dataclass
class Global:
    project: str
    domain: str


@dataclass
class StreamlitConfig:
    resources: Resources = field(default_factory=lambda: Resources(cpu="3", mem="4Gi"))


@dataclass
class LLMConfig:
    models: list[Model]
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    cache_workflow: Optional[CacheWorkflow] = None
    global_config: Optional[Global] = None


def get_config() -> LLMConfig:
    from mashumaro.codecs.yaml import YAMLDecoder

    config_file = os.getenv("LLM_CONFIG_FILE")
    if config_file is None:
        msg = "LLM_CONFIG_FILE must be set"
        raise RuntimeError(msg)

    decoder = YAMLDecoder(LLMConfig)

    with open(config_file) as f:
        config = decoder.decode(f.read())
    return config
