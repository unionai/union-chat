# Union LLM Serving

## Deploying to a Union tenant

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Cache models: `uv run python cache_models_wf.py config_remote.yaml`
3. Deploy chatbot endpoint that depends on model: `uv run python deploy_app.py config_remote.yaml`

## Adding a New model

Update `config_remote.yaml` with a new model. For new models, you may need to update the VLLM or SGLang image.

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
