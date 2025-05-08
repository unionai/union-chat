import os
from typing import Optional
from dataclasses import dataclass, field
from union import Resources
from flytekit.extras.accelerators import GPUAccelerator


PLACEHOLDER_API_KEY = "PLACEHOLDER_API_KEY"


@dataclass
class LLMRuntime:
    resources: Resources
    stream_model: bool
    llm_type: str
    scaledown_after: int = 300
    image: str | None = None
    accelerator: Optional[str] = None
    extra_args: str = ""
    app_kwargs: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    min_replicas: int = 0

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
    local: bool = False
    secret_key: Optional[str] = None

    def get_endpoint_env_var(self, i: int) -> str:
        return f"ENDPOINT_{i}"
    
    def get_public_endpoint_env_var(self, i: int) -> str:
        return f"PUBLIC_ENDPOINT_{i}"

    def get_endpoint(self, i: int) -> str:
        if self.base_url is not None:
            return self.base_url

        endpoint = os.getenv(self.get_endpoint_env_var(i))
        if endpoint is None:
            msg = "base_url_env_var is not set"
            raise RuntimeError(msg)
        return endpoint
    
    def get_public_endpoint(self, i: int) -> str:
        endpoint = os.getenv(self.get_public_endpoint_env_var(i))
        if endpoint is None:
            msg = "public_endpoint_env_var is not set"
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
class SpecialMessage:
    title: str
    body: list[str]


@dataclass
class StreamlitConfig:
    subdomain: Optional[str] = None
    resources: Resources = field(default_factory=lambda: Resources(cpu="3", mem="4Gi"))
    show_api_keys: bool = False
    show_api_keys_message: str = ""
    special_message: Optional[SpecialMessage] = None


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
