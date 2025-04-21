from union.app.llm import VLLMApp
from union import ImageSpec
from models import get_config
from flytekit.extras.accelerators import GPUAccelerator


config = get_config()

llm_apps = {}

vllm_image = ImageSpec(
    name="serving-vllm",
    packages=["union[vllm]==0.1.173"],
    registry="ghcr.io/unionai-oss",
)

for model_config in config.models:
    if model_config.llm_runtime is None:
        raise ValueError("llm_runtime must not be None")
    if model_config.model_uri is None:
        raise ValueError("model_uri must be defined")

    if model_config.llm_runtime.accelerator is None:
        raise ValueError("accelerator must be set")

    if model_config.app_name is None:
        raise ValueError("app_name must be defined")

    llm = VLLMApp(
        name=model_config.app_name,
        container_image=vllm_image,
        requests=model_config.llm_runtime.resources,
        limits=model_config.llm_runtime.resources,
        port=8080,
        model_id=model_config.model_id,
        model=model_config.model_uri,
        stream_model=model_config.llm_runtime.stream_model,
        accelerator=GPUAccelerator(model_config.llm_runtime.accelerator),
        scaledown_after=300,
    )
