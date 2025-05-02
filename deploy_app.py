from typing import Optional
from union.app.llm import VLLMApp, SGLangApp
from union.app import App, Input
from union import ImageSpec
from models import get_config_from_file
from flytekit.extras.accelerators import GPUAccelerator
from union.remote import UnionRemote
from union import Artifact
import click


@click.command()
@click.argument("config_file")
@click.option("--model", help="Only deploy a model")
def main(config_file: str, model: Optional[str]):
    config = get_config_from_file(config_file)

    llm_apps = {}
    llm_env_vars = {}
    seen_env_vars = set()

    assert config.global_config is not None

    for i, model_config in enumerate(config.models):
        if model_config.llm_runtime is None:
            raise ValueError("llm_runtime must not be None")

        if model_config.llm_runtime.accelerator is None:
            raise ValueError("accelerator must be set")

        if model_config.name is None:
            raise ValueError("name must be defined")

        if not model_config.model_uri:
            model_artifact = Artifact(name=model_config.name).query(
                project=config.global_config.project,
                domain=config.global_config.domain,
            )
        else:
            model_artifact = model_config.model_uri

        llm_type = model_config.llm_runtime.llm_type
        if llm_type == "vllm":
            LLMCls = VLLMApp
            port = 8000
        else:
            LLMCls = SGLangApp
            port = 8080

        extra_args = " ".join([model_config.llm_runtime.extra_args, "--api-key ABC"])
        llm = LLMCls(
            name=f"{model_config.name}-{llm_type}",
            container_image=model_config.llm_runtime.image,
            requests=model_config.llm_runtime.resources,
            limits=model_config.llm_runtime.resources,
            port=port,
            model_id=model_config.model_id,
            model=model_artifact,
            stream_model=model_config.llm_runtime.stream_model,
            accelerator=GPUAccelerator(model_config.llm_runtime.accelerator),
            scaledown_after=300,
            extra_args=extra_args,
            env=model_config.llm_runtime.env,
            # we'll authenticate not via Union's auth, but via the API key
            requires_auth=False,
        )
        llm_apps[model_config.name] = llm
        base_url_env_var = model_config.get_endpoint_env_var(i)

        if base_url_env_var in seen_env_vars:
            raise ValueError("base_url_env_bar must be unique")

        seen_env_vars.add(base_url_env_var)
        llm_env_vars[model_config.name] = base_url_env_var

    remote = UnionRemote(
        default_domain=config.global_config.domain,
        default_project=config.global_config.project,
    )
    if model is not None:
        print(f"Deploying only {model}")
        remote.deploy_app(llm_apps[model])
        return

    streamlit_image = ImageSpec(
        name="streamlit-chat",
        packages=[
            "streamlit==1.44.1",
            "openai==1.75.0",
            "mashumaro[yaml]==3.15",
            "union-runtime==0.1.17",
            "union==0.1.173",
        ],
        builder="union",
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
        include=["chatapp.py", "models.py", "config_remote.yaml", "pyproject.toml"],
        dependencies=list(llm_apps.values()),
        env={"LLM_CONFIG_FILE": "config_remote.yaml"},
        requests=config.streamlit.resources,
        limits=config.streamlit.resources,
    )

    remote.deploy_app(streamlit_app)


if __name__ == "__main__":
    main()
