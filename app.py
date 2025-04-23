from union.app.llm import VLLMApp, SGLangApp
from union.app import App, Input
from union import ImageSpec
from models import get_config
from flytekit.extras.accelerators import GPUAccelerator


config = get_config()

llm_apps = {}
llm_env_vars = {}
seen_env_vars = set()

for model_config in config.models:
    if model_config.llm_runtime is None:
        raise ValueError("llm_runtime must not be None")
    if model_config.model_uri is None:
        raise ValueError("model_uri must be defined")

    if model_config.llm_runtime.accelerator is None:
        raise ValueError("accelerator must be set")

    if model_config.name is None:
        raise ValueError("name must be defined")

    if model_config.llm_runtime.llm_type == "VLLM":
        LLMCls = VLLMApp
        image = "ghcr.io/unionai-oss/serving-vllm:0.1.17"
    else:
        # Add support for SGLang
        assert False
        LLMCls = SGLangApp
        # Update to using
        image = "ghcr.io/unionai-oss/serving-vllm:0.1.17"

    llm = LLMCls(
        name=model_config.name,
        container_image=image,
        requests=model_config.llm_runtime.resources,
        limits=model_config.llm_runtime.resources,
        port=8080,
        model_id=model_config.model_id,
        model=model_config.model_uri,
        stream_model=model_config.llm_runtime.stream_model,
        accelerator=GPUAccelerator(model_config.llm_runtime.accelerator),
        scaledown_after=300,
        extra_args=model_config.llm_runtime.extra_args,
    )
    llm_apps[model_config.name] = llm
    if model_config.base_url_env_var is None:
        raise ValueError("base_url_env_var must be set")

    if model_config.base_url_env_var in seen_env_vars:
        raise ValueError("base_url_env_bar must be unique")

    seen_env_vars.add(model_config.base_url_env_var)
    llm_env_vars[model_config.name] = model_config.base_url_env_var

streamlit_image = ImageSpec(
    name="streamlit-chat",
    packages=[
        "streamlit==1.44.1",
        "openai==1.75.0",
        "mashumaro[yaml]==3.15",
        "union-runtime==0.1.17",
        "union==0.1.173",
    ],
    registry="ghcr.io/unionai-oss",
)

streamlit_app = App(
    name="union-llm-serving",
    container_image=streamlit_image,
    inputs=[
        Input(
            name=name,
            value=llm_app.query_endpoint(),
            env_var=llm_env_vars[name],
            download=False,
        )
        for name, llm_app in llm_apps.items()
    ],
    port=8501,
    args="streamlit run chatapp.py",
    include=["chatapp.py", "models.py", "config_remote.yaml"],
    dependencies=list(llm_apps.values()),
    env={"LLM_CONFIG_FILE": "config_remote.yaml"},
    requests=config.streamlit.resources,
)
