import os
from typing import Optional
from dataclasses import dataclass, field
from union import Resources
from flytekit.extras.accelerators import GPUAccelerator


@dataclass
class LLMRuntime:
    image: str
    resources: Resources
    accelerator: Optional[str]
    stream_model: bool
    llm_type: str
    extra_args: str = ""
    app_kwargs: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass
class Model:
    display_name: str
    model_id: str
    name: Optional[str] = None
    model_uri: Optional[str] = None
    base_url: Optional[str] = None
    llm_runtime: Optional[LLMRuntime] = None
    cache_version: str = "v1"
    max_tokens: Optional[int] = None

    def get_endpoint_env_var(self, i: int) -> str:
        return f"ENDPOINT_{i}"

    def get_endpoint(self, i: int) -> str:
        if self.base_url is not None:
            return self.base_url

        endpoint = os.getenv(self.get_endpoint_env_var(i))
        if endpoint is None:
            msg = "base_url_end_var is not set"
            raise RuntimeError(msg)
        return endpoint


@dataclass
class CacheWorkflow:
    hf_secret_key: str
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
    config_file = os.getenv("LLM_CONFIG_FILE")
    if config_file is None:
        msg = "LLM_CONFIG_FILE must be set"
        raise RuntimeError(msg)

    return get_config_from_file(config_file)


def get_config_from_file(file: str) -> LLMConfig:
    from mashumaro.codecs.yaml import YAMLDecoder

    decoder = YAMLDecoder(LLMConfig)

    with open(file) as f:
        config = decoder.decode(f.read())
    return config
