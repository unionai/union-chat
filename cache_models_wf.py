import os

import hashlib
import sys
from functools import partial
from typing import Generator
from flytekit import Cache
from flytekit import Workflow, Secret
from models import get_config_from_file
from dataclasses import dataclass
import click
from union import (
    task,
    ImageSpec,
    current_context,
    UnionRemote,
    FlyteDirectory,
    Artifact,
)
from flytekit.core.context_manager import ExecutionParameters
from flytekit.tools.fast_registration import FastPackageOptions, CopyFileDetection

hf_cache_image = ImageSpec(
    name="hfhub-cache",
    packages=["union==0.1.176", "huggingface-hub[hf_transfer]==0.30.2"],
    builder="union",
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)


@task
def validate_repo(hf_repo: str, hf_secret_key: str) -> str:
    from huggingface_hub import list_repo_commits, repo_exists

    ctx = current_context()
    token = ctx.secrets.get(key=hf_secret_key)

    if not repo_exists(hf_repo, token=token):
        raise ValueError(f"Huggingface repo: {hf_repo} does not exist")

    commit = list_repo_commits(hf_repo, token=token)[0]
    return commit.commit_id


def _yield_files(hfs, repo_id: str, revision: str) -> Generator[dict, None, None]:
    for _, _, files in hfs.walk(repo_id, revision=revision, detail=True):
        for file_details in files.values():
            yield file_details


def _stream_file_to_dir(
    file_details: dict,
    hfs,
    prefix_len: int,
    directory: FlyteDirectory,
    chunk_size: int,
):
    name = file_details["name"]
    size = file_details["size"]
    with hfs.open(name, "rb", block_size=0) as res:
        filename = name[prefix_len:]
        ff = directory.new_file(filename)
        copied = 0
        print(
            f"Copying {name} to {ff.path}, size: {size}. Total chunks: {size // chunk_size}",
            flush=True,
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
                    print(f"Completed copying {percent_complete} %...", flush=True)
        print(f"Copied {name} to {ff.path}", flush=True)


def stream_all_files_to_flytedir(
    repo_id: str,
    commit: str,
    chunk_size: int,
    token: str,
) -> FlyteDirectory:
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

    directory = FlyteDirectory.new_remote()

    hfs = HfFileSystem(token=token)

    root_name_detail = hfs.info(repo_id, revision=commit)
    prefix = root_name_detail["name"]
    prefix_len = len(prefix) + 1

    stream_file_partial = partial(
        _stream_file_to_dir,
        hfs=hfs,
        prefix_len=prefix_len,
        directory=directory,
        chunk_size=chunk_size,
    )

    for file_details in _yield_files(hfs, repo_id=repo_id, revision=commit):
        stream_file_partial(file_details)

    return directory


def _get_remote(ctx: ExecutionParameters) -> UnionRemote:
    """
    Get the remote object for the current execution. This is used to interact with the Union backend.
    Args:
        self: flytekit.core.context_manager.ExecutionParameters
    Returns: UnionRemote
    """
    project = ctx.execution_id.project if ctx.execution_id else None
    domain = ctx.execution_id.domain if ctx.execution_id else None
    raw_output = ctx.raw_output_prefix
    return UnionRemote(
        config=None, project=project, domain=domain, data_upload_location=raw_output
    )


def _emit_artifact(ctx: ExecutionParameters, o: Artifact) -> Artifact:
    """
    Emit an artifact to Union. This will create a new artifact with the given name and version and will
    associate with this execution.
    If o is None or not an Artifact, this function will do nothing.
    Args:
        self: flytekit.core.context_manager.ExecutionParameters
        o: Artifact

    Raises: Exception if artifact creation fails.
    """
    # TODO add node_id to the context.
    from union.internal.artifacts import artifacts_pb2

    # Emit artifact
    if "HOSTNAME" in os.environ:
        hostname = os.environ["HOSTNAME"]
        try:
            node_id = hostname.split("-")[1]
        except Exception:
            node_id = "n1"
    else:
        node_id = "n1"

    o.set_source(
        artifacts_pb2.ArtifactSource(
            workflow_execution=ctx.execution_id.to_flyte_idl(),
            task_id=ctx.task_id.to_flyte_idl(),
            retry_attempt=int(os.getenv("FLYTE_ATTEMPT_NUMBER", "0")),
            node_id=node_id,
        )
    )
    remote = _get_remote(ctx)
    try:
        return remote.create_artifact(o)
    except Exception as e:
        print(f"Failed to create artifact {o}: {e}")
        return remote.get_artifact(query=o.query().to_flyte_idl())


@dataclass
class ArtifactInfo:
    artifact_name: str
    blob: str
    model_uri: str


@task
def cache_model_from_hf(
    hf_repo: str,
    artifact_name: str,
    commit: str,
    chunk_size: int,
    hf_secret_key: str,
) -> ArtifactInfo:
    print(
        f"Caching model from huggingface repo: {hf_repo}, commit: {commit}",
        flush=True,
    )
    ctx = current_context()
    token = ctx.secrets.get(key=hf_secret_key)

    directory = stream_all_files_to_flytedir(hf_repo, commit, chunk_size, token)
    print(f"Data streamed to {directory.path}")

    o = Artifact(
        name=artifact_name,
        python_type=FlyteDirectory,
        python_val=directory,
        short_description=f"Model cached from huggingface repo: {hf_repo}, commit: {commit} "
        f"by execution: {ctx.execution_id}.",
        project=ctx.execution_id.project if ctx.execution_id else None,
        domain=ctx.execution_id.domain if ctx.execution_id else None,
    )
    print(f"Emitting artifact, {o}")
    a: Artifact = _emit_artifact(ctx, o)
    return ArtifactInfo(
        artifact_name=artifact_name,
        blob=directory.path,
        model_uri=a.metadata().uri if a else "NA",
    )


def hash_current_file(config_file: str) -> str:
    current_file = sys.argv[0]

    m = hashlib.sha1()
    with open(current_file, "rb") as f:
        m.update(f.read())

    with open(config_file, "rb") as f:
        m.update(f.read())

    return m.hexdigest()


@click.command()
@click.argument("config_file")
def main(config_file: str):
    config = get_config_from_file(config_file)

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

    hf_secret = Secret(key=cache_workflow.hf_secret_key)
    union_secret = Secret(key="EAGER_API_KEY", env_var="UNION_API_KEY")
    caches = []

    name_to_model_id = {}
    for i, model in enumerate(config.models):
        if (
            model.local
            or model.llm_runtime.llm_type == "openai"
            or model.llm_runtime.llm_type == "ollama"
        ):
            continue
        var_name = f"hf_repo_{i}"
        name_to_model_id[var_name] = model.model_id

        imperative_wf.add_workflow_input(var_name, str)

        cache = Cache(version=model.cache_version)
        caches.append(cache)

        validate_repo_task = task(
            container_image=hf_cache_image,
            secret_requests=[hf_secret],
            cache=cache,
        )(validate_repo.task_function)

        validate_repo_node = imperative_wf.add_entity(
            validate_repo_task,
            hf_repo=imperative_wf.inputs[var_name],
            hf_secret_key=cache_workflow.hf_secret_key,
        )

        cache_model_from_hf_task = task(
            container_image=hf_cache_image,
            secret_requests=[hf_secret, union_secret],
            requests=cache_workflow.resources,
            limits=cache_workflow.resources,
            cache=cache,
        )(cache_model_from_hf.task_function)

        cache_model_from_hf_node = imperative_wf.add_entity(
            cache_model_from_hf_task,
            hf_repo=imperative_wf.inputs[var_name],
            artifact_name=model.name,
            hf_secret_key=cache_workflow.hf_secret_key,
            commit=validate_repo_node.outputs["o0"],
            chunk_size=cache_workflow.chunk_size,
        )

    version = hash_current_file(config_file=config_file)

    wf = remote.register_script(
        imperative_wf,
        source_path=os.getcwd(),
        version=version,
        fast_package_options=FastPackageOptions(
            ignores=[],
            copy_style=CopyFileDetection.LOADED_MODULES,
        ),
    )

    execution = remote.execute(wf, inputs=name_to_model_id)
    print(execution.execution_url)


if __name__ == "__main__":
    main()
