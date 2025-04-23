from union.app.llm import VLLMApp, SGLangApp
from union.app import App, Input
from union import ImageSpec
from models import get_config
from flytekit.extras.accelerators import GPUAccelerator
from union.remote import UnionRemote


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

    env = {}
    llm_type = model_config.llm_runtime.llm_type
    if llm_type == "vllm":
        LLMCls = VLLMApp
        image = "ghcr.io/unionai-oss/serving-vllm:0.1.17"
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        port = 8000
    else:
        LLMCls = SGLangApp
        image = "ghcr.io/unionai-oss/serving-sglang:0.1.17"
        port = 8080

    llm = LLMCls(
        name=f"{model_config.name}-{llm_type}",
        container_image=image,
        requests=model_config.llm_runtime.resources,
        limits=model_config.llm_runtime.resources,
        port=port,
        model_id=model_config.model_id,
        model=model_config.model_uri,
        stream_model=model_config.llm_runtime.stream_model,
        accelerator=GPUAccelerator(model_config.llm_runtime.accelerator),
        scaledown_after=300,
        extra_args=model_config.llm_runtime.extra_args,
        env=env,
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
    name="union-llm-serving-2",
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
    limits=config.streamlit.resources,
)

if __name__ == "__main__":
    assert config.global_config is not None
    remote = UnionRemote(
        default_domain=config.global_config.domain,
        default_project=config.global_config.project,
    )
    remote.deploy_app(streamlit_app)
