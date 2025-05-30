import click
import union

from dataclasses import dataclass
from typing import Optional
from union.app.llm import VLLMApp, SGLangApp
from union.app import App, Input
from union import ImageSpec
from models import get_config_from_file, PLACEHOLDER_API_KEY, LLMConfig
from flytekit.extras.accelerators import GPUAccelerator
from union.remote import UnionRemote
from union import Artifact
from ollama_app import OllamaApp, ollama_image


@dataclass
class LLMDeployment:
    llm_apps: dict[str, App]
    llm_env_vars: dict[str, str]
    public_url_env_vars: dict[str, str]
    app_secrets: list[union.Secret]


def deploy_llm_endpoints(
    config: LLMConfig,
    model: Optional[str],
    remote: UnionRemote,
) -> LLMDeployment:
    llm_apps = {}
    llm_env_vars = {}
    seen_env_vars = set()

    # Deploy the LLM endpoints
    app_secrets = []
    for i, model_config in enumerate(config.models):

        if model_config.local:
            continue

        if model_config.llm_runtime is None:
            raise ValueError("llm_runtime must not be None")

        if model_config.name is None:
            raise ValueError("name must be defined")

        if not model_config.model_uri:
            model_artifact = Artifact(name=model_config.name).query(
                project=config.global_config.project,
                domain=config.global_config.domain,
            )
        elif llm_type == "ollama":
            model_artifact = None
        else:
            model_artifact = model_config.model_uri

        llm_type = model_config.llm_runtime.llm_type
        if llm_type == "vllm":
            LLMCls = VLLMApp
            port = 8000
            image = model_config.llm_runtime.image
        elif llm_type == "sglang":
            LLMCls = SGLangApp
            port = 8080
            image = model_config.llm_runtime.image
        elif llm_type == "ollama":
            LLMCls = OllamaApp
            port = 11434
            image = ollama_image
        elif llm_type == "openai":
            LLMCls = None
            port = None
            image = None
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

        if model_config.secret_key is not None:
            _secret_arg = model_config.get_secret_env_var(i)
            secret = union.Secret(key=model_config.secret_key, env_var=_secret_arg)
            app_secrets.append(secret)
            secrets, secret_arg = [secret], f"${_secret_arg}"
        else:
            secrets = None
            secret_arg = PLACEHOLDER_API_KEY

        extra_args = " ".join([
            model_config.llm_runtime.extra_args,
            f"--api-key {secret_arg}",
        ])

        kwargs = {}
        if model_config.llm_runtime.accelerator is not None:
            kwargs["accelerator"] = GPUAccelerator(model_config.llm_runtime.accelerator)

        if model_config.llm_runtime.llm_type == "openai":
            # don't deploy the app for openai models
            llm_app = None
            base_url_env_var = None
        else:
            llm_app = LLMCls(
                name=f"{model_config.name}-{llm_type}",
                container_image=image,
                requests=model_config.llm_runtime.resources,
                limits=model_config.llm_runtime.resources,
                port=port,
                model_id=model_config.model_id,
                model=model_artifact,
                min_replicas=model_config.llm_runtime.min_replicas,
                stream_model=model_config.llm_runtime.stream_model,
                scaledown_after=model_config.llm_runtime.scaledown_after,
                extra_args=extra_args,
                env=model_config.llm_runtime.env,
                secrets=secrets,
                # we'll authenticate not via Union's auth, but via the --api-key flag
                requires_auth=False,
                **kwargs,
            )
            base_url_env_var = model_config.get_endpoint_env_var(i)
            if base_url_env_var in seen_env_vars:
                raise ValueError("base_url_env_bar must be unique")

            seen_env_vars.add(base_url_env_var)
            llm_env_vars[model_config.name] = base_url_env_var
            llm_apps[model_config.name] = llm_app

    if model is not None:
        print(f"Deploying only {model}")
        remote.deploy_app(llm_apps[model])
        return
    
    public_url_env_vars = {}
    for i, app in enumerate(llm_apps.values()):
        app_idl = remote.deploy_app(app)
        public_url_env_vars[model_config.get_public_endpoint_env_var(i)] = app_idl.status.ingress.public_url

    return LLMDeployment(
        llm_apps=llm_apps,
        llm_env_vars=llm_env_vars,
        public_url_env_vars=public_url_env_vars,
        app_secrets=app_secrets,
    )


def deploy_streamlit_app(
    config: LLMConfig,
    llm_deployment: LLMDeployment,
    remote: UnionRemote,
):

    streamlit_image = ImageSpec(
        name="streamlit-union-chat",
        packages=[
            "fastmcp==2.3.4",
            "langchain-core==0.3.60",
            "nest-asyncio==1.6.0",
            "streamlit==1.44.1",
            "streamlit-local-storage==0.0.25",
            "openai==1.75.0",
            "mashumaro[yaml]==3.15",
            "union-runtime==0.1.18",
            "union==0.1.181",
        ],
        builder="union",
    )

    streamlit_app = App(
        name="union-llm-serving-2",
        type="Streamlit",
        container_image=streamlit_image,
        inputs=[
            Input(
                name=name,
                value=llm_app.query_endpoint(),
                env_var=llm_deployment.llm_env_vars[name],
                download=False,
            )
            for name, llm_app in llm_deployment.llm_apps.items()
        ],
        port=8501,
        include=[
            "chatapp.py",
            "app_utils.py",
            "models.py",
            "mcp_client.py",
            "config_remote.yaml",
            "pyproject.toml",
        ],
        secrets=llm_deployment.app_secrets,
        min_replicas=1,
        max_replicas=3,
        args="streamlit run chatapp.py",
        dependencies=list(llm_deployment.llm_apps.values()),
        env={
            "LLM_CONFIG_FILE": "config_remote.yaml",
            **llm_deployment.public_url_env_vars,
        },
        requests=config.streamlit.resources,
        limits=config.streamlit.resources,
        subdomain=config.streamlit.subdomain,
    )

    remote.deploy_app(streamlit_app)


@click.command()
@click.argument("config_file")
@click.option("--model", help="Only deploy a model")
def main(config_file: str, model: Optional[str]):
    config = get_config_from_file(config_file)
    assert config.global_config is not None

    remote = UnionRemote(
        default_domain=config.global_config.domain,
        default_project=config.global_config.project,
    )

    llm_deployment = deploy_llm_endpoints(config, model, remote)
    deploy_streamlit_app(config, llm_deployment, remote)


if __name__ == "__main__":
    main()
