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

## Securing your LLM endpoints

The default deployment will use a placeholder API key to authenticate requests
to the remote LLM endpoints. To supply your own API keys, do the following:

```bash
union create secret --project <my_project> --domain <my_domain> --name model-api-key
```

Supply the secret value in the CLI input, which will create a secret called
`model-api-key`. Update the `config_remote.yaml` file like so:

```yaml
...

models:
  # update the secret_key value for all applicable models.
  - display_name: Llama-3.2-1B-Instruct
    ...
    secret_key: model-api-key
  - ...
```

Then redeploy the apps:

```bash
uv run python deploy_app.py config_remote.yaml
```

## Local Development

1. Install Ollama: https://ollama.com/download and start it
2. Download models that are already defined in `config_local.yaml`

```bash
ollama pull gemma3:1b
ollama pull qwen2.5-coder:0.5b
```

3. Run chatapp:

```bash
LLM_CONFIG_FILE=config_local.yaml uv run streamlit run chatapp.py
```
