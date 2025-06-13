import click

import union
from union.app import App
from union import UnionRemote
from models import LLMConfig, get_config_from_file


def deploy_mcp_app(config: LLMConfig, remote: UnionRemote) -> App:
    image = union.ImageSpec(
        name="mcp-server",
        apt_packages=["git"],
        packages=["uv", "union-runtime>=0.1.17", "fastmcp"],
        builder="union",
    )

    app = App(
        name="union-mcp-test-2",
        type="MCP Server",
        port=8000,
        include=["mcp_app.py"],
        container_image=image,
        args="mcp run mcp_app.py --transport sse",
        requests=union.Resources(cpu=2, mem="1Gi"),
        min_replicas=0,
        requires_auth=False,
    )

    remote.deploy_app(app)
    return app


@click.command()
@click.argument("config_file")
def main(config_file: str):
    config = get_config_from_file(config_file)
    assert config.global_config is not None

    remote = UnionRemote(
        default_domain=config.global_config.domain,
        default_project=config.global_config.project,
    )

    deploy_mcp_app(config, remote)


if __name__ == "__main__":
    main()
