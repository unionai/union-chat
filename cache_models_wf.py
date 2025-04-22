from typing import Annotated
from typing import Optional
from union import task, ImageSpec, current_context, UnionRemote, FlyteDirectory
from flytekit.tools.fast_registration import FastPackageOptions, CopyFileDetection

hf_cache_image = ImageSpec(
    name="hfhub-cache",
    packages=["union==0.1.176", "huggingface-hub[hf_transfer]==0.30.2"],
    registry="ghcr.io/unionai-oss",
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)


@task
def validate_repo(hf_repo: str, hf_token_secret_key: str) -> str:
    from huggingface_hub import list_repo_commits, repo_exists

    ctx = current_context()
    token = ctx.secrets.get(key=hf_token_secret_key)

    if not repo_exists(hf_repo, token=token):
        raise ValueError(f"Huggingface repo: {hf_repo} does not exist")

    commit = list_repo_commits(hf_repo, token=token)[0]
    return commit.commit_id


def stream_all_files_to_flytedir(
    repo_id: str,
    commit: str,
    chunk_size: int,
    token: str | None = None,
) -> tuple[FlyteDirectory, str | None]:
    """
    TODO we should use hf-transfer for this, but the only option on hf-transfer is to download the files to local disk.
    Stream all files in a Hugging Face Hub repository to a FlyteDirectory.

    Args:
        :param repo_id: str The repository ID (e.g., 'julien-c/EsperBERTo-small').
        :param commit: str The commit ID.
        :param token: str[optional] The Hugging Face Hub token for authentication.
        :param chunk_size: int[optional] The chunk size to use when streaming the model files.
    """
    from huggingface_hub import HfFileSystem

    directory = union.FlyteDirectory.new_remote()
    card = None

    hfs = HfFileSystem(token=token)
    for file_details in hfs.ls(repo_id, revision=commit, detail=True):
        f = file_details["name"]
        if f.endswith(".md"):
            print(f"Reading card from {f}")
            with hfs.open(f, "r") as res:
                card = res.read()
            continue

        with hfs.open(f, "rb", block_size=0) as res:
            filename = os.path.basename(f)
            size = file_details["size"]
            ff = directory.new_file(filename)
            copied = 0
            print(
                f"Copying {f} to {ff.path}, size: {size}. Total chunks: {size // chunk_size}"
            )
            with ff.open("wb") as sink:
                while True:
                    chunk = res.read(chunk_size)
                    sink.write(chunk)
                    copied = copied + len(chunk)
                    if copied >= size:
                        break
                    percent_complete = copied / size * 100
                    if int(percent_complete) > 0 and int(percent_complete) % 10 == 0:
                        print(f"Completed copying {percent_complete} %...")
        print(f"Copied {f} to {directory.path}")
    return directory, card


@task
def cache_model_from_hf(hf_repo: str, commit: str, chunk_size: int, hf_token_key: str):
    print(
        f"Caching model from huggingface repo: {hf_repo}, commit: {commit}",
        flush=True,
    )
    ctx = current_context()
    token = ctx.secrets.get(key=hf_token_key)

    directory, card = stream_all_files_to_flytedir(hf_repo, commit, chunk_size, token)


# @task(container_image=hf_cache_image)
# def see_hf_hub() -> str:
#     import os

#     return os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "unknown")

# from union import Artifact
from flytekit import Workflow, Secret
from models import get_config


if __name__ == "__main__":
    # Validate config
    config = get_config()
    if config.global_config is None:
        raise ValueError("global_config must be set")

    cache_workflow = config.cache_workflow
    if cache_workflow is None:
        raise ValueError("cache_workflow config must be set")

    imperative_wf = Workflow(name="cache_models_wf")

    remote = UnionRemote(
        default_domain=config.global_config.domain,
        default_project=config.global_config.project,
    )
    imperative_wf.add_workflow_input("hf_repo", str)
    imperative_wf.add_workflow_input("hf_token_secret_key", str)
    imperative_wf.add_workflow_input("chunk_size", int)

    # MyArtifact = Artifact(name="sample-artifact")
    # return_type = fun.task_function.__annotations__["return"]
    # fun.task_function.__annotations__["return"] = Annotated[return_type, MyArtifact]

    validate_repo_task = task(
        container_image=hf_cache_image,
        secret_requests=[Secret(key=cache_workflow.secret_key)],
        # requests=cache_workflow.resources,
        # limits=cache_workflow.resources,
        # accelerator=cache_workflow.accelerator_obj,
    )(validate_repo)

    validate_repo_node = imperative_wf.add_entity(
        validate_repo_task,
        hf_repo=imperative_wf.inputs["hf_repo"],
        hf_token_secret_key=imperative_wf.inputs["hf_token_secret_key"],
    )

    imperative_wf.add_workflow_output("info", validate_repo_node.outputs["o0"])

    import os

    wf = remote.register_script(
        imperative_wf,
        source_path=os.getcwd(),
        fast_package_options=FastPackageOptions(
            ignores=[],
            copy_style=CopyFileDetection.LOADED_MODULES,
        ),
    )

    execution = remote.execute(
        wf,
        inputs={
            "hf_repo": "microsoft/Phi-3.5-mini-instruct",
            "hf_token_secret_key": "thomasjpfan-hugging-face",
            "chunk_size": 8 * 1024 * 1024,
        },
    )
    print(execution.execution_url)
