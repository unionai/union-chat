from dataclasses import dataclass
from union import ImageSpec
from union.app import App


ollama_image = ImageSpec(
    name="ollama-app",
    apt_packages=["curl"],
    packages=[
        "union-runtime==0.1.18",
    ],
    commands=[
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
    ],
    builder="union",
)


@dataclass
class OllamaApp(App):

    port: int = 11434
    type: str = "Ollama"
    extra_args: str | list[str] = ""
    model_id: str = ""
    model: str | None = None
    stream_model: bool = False

    def __post_init__(self):
        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.args:
            raise ValueError("args can not be set for OllamaApp. Use `extra_args` to add extra arguments to SGLang")

        self.args = f"ollama serve & sleep 2 && ollama pull {self.model_id} && tail -f /dev/null"

        if self.inputs:
            raise ValueError("inputs can not be set for OllamaApp")

        self.env["OLLAMA_HOST"] = "0.0.0.0"
        self.env["OLLAMA_ORIGINS"] = "*"
        self.env["OLLAMA_MODELS"] = "/home/union/.ollama/models"
        super().__post_init__()


if __name__ == "__main__":
    app = OllamaApp(model_id="qwen2.5:0.5b")
