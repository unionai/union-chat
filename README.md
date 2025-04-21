# Union LLM Serving

## Deploying to a Union tenant

1.

## Adding a New Model

## Local Development

1. Install Ollama: https://ollama.com/download and start it

2. Download models that are already defined in `config_local.yaml`

```bash
ollama pull gemma3:1b
ollama pull qwen2.5-coder:0.5b
```

2. Run chatapp:

```bash
CONFIG_FILE=config_local.yaml uv run streamlit run chatapp.py
```
