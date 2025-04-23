from flytekit import task, Workflow, dynamic, workflow, Cache, ImageSpec, map_task

from flytekit.tools.fast_registration import FastPackageOptions, CopyFileDetection

hf_cache_image = ImageSpec(
    name="hfhub-cache",
    packages=["union==0.1.176", "huggingface-hub[hf_transfer]==0.30.2"],
    registry="ghcr.io/unionai-oss",
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)


# @task(cache=Cache(version="v1"), container_image=hf_cache_image)
@task(container_image=hf_cache_image)
def do_something(x: int) -> int:
    return x + 1


if __name__ == "__main__":
    wf1 = Workflow(name="single")
    wf1.add_workflow_input("x", int)
    wf1.add_workflow_input("y", int)

    wf_entity1 = wf1.add_entity(do_something, x=wf1.inputs["x"])
    wf_entity2 = wf1.add_entity(do_something, x=wf1.inputs["y"])
    wf1.add_workflow_output("out", wf_entity1.outputs["o0"])

    # imperative_wf.add_workflow_input("items")

    from union.remote import UnionRemote
    import os

    remote = UnionRemote(default_domain="development", default_project="thomasjpfan")
    wf = remote.register_script(
        wf1,
        source_path=os.getcwd(),
        fast_package_options=FastPackageOptions(
            ignores=[],
            copy_style=CopyFileDetection.LOADED_MODULES,
        ),
    )

    execution = remote.execute(wf, inputs={"x": 5, "y": 4})
    print(execution.execution_url)
