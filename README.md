# Union LLM Serving

## Setup

Copy and customize the remote configuration:

```bash
cp config_remote.yaml.template config_remote.yaml
```

Update the following values:

```yaml
cache_workflow:
  hf_secret_key: huggingface-api-key  # update this with the secret key for your huggingface api key

global_config:
  project: <project_name>  # change the project name here
  domain: development  # update this to your desired domain
```

## Deploying to a Union tenant

Install `uv`: https://docs.astral.sh/uv/getting-started/installation/

Then cache models:

```bash
uv run python cache_models_wf.py config_remote.yaml
```

Deploy chatbot endpoint that depends on model:

```bash
uv run python deploy_app.py config_remote.yaml
```

## Adding a New model

Update `config_remote.yaml` with a new model. For new models, you may need to update the vLLM or SGLang image.

## Local Development

1. Install Ollama: https://ollama.com/download and start it
2. Download models that are already defined in `config_local.yaml`

```bash
ollama pull gemma3:1b
ollama pull qwen2.5-coder:0.5b
```

3. Run chatapp:

```bash
uv run python deploy_app.py config_local.yaml
```
