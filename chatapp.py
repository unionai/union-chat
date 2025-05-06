import tomllib
import logging
import os

from dataclasses import dataclass
from models import get_config, PLACEHOLDER_API_KEY
from app_utils import wake_up_endpoints

import streamlit as st
from openai import OpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define types
@dataclass
class ClientInfo:
    client: OpenAI
    model_id: str
    endpoint: str
    max_tokens: int | None = None
    local: bool = False


@dataclass
class InferenceSettings:
    model: str
    max_output_tokens: int | None
    temperature: float
    top_p: float


@st.cache_resource
def load_client_infos() -> dict[str, ClientInfo]:
    config = get_config()

    client_infos = {}
    remote_endpoints, headers = [], []
    for i, model_config in enumerate(config.models):
        endpoint = model_config.get_endpoint(i)
        if model_config.local:
            api_key = "ollama"
        else:
            api_key = os.getenv("UNION_ENDPOINT_SECRET", PLACEHOLDER_API_KEY)
            remote_endpoints.append(endpoint)
            headers.append({"Authorization": f"Bearer {api_key}"})

        client_infos[model_config.display_name] = ClientInfo(
            client=OpenAI(base_url=f"{endpoint}/v1", api_key=api_key),
            model_id=model_config.model_id,
            endpoint=endpoint,
            max_tokens=model_config.max_tokens,
            local=model_config.local,
        )

    return client_infos, remote_endpoints, headers


def init_session_state():
    # initialize session state
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = None
    if "messages" not in st.session_state:
        clear_chat()
    if "selected_prewritten_prompt" not in st.session_state:
        st.session_state["selected_prewritten_prompt"] = None


def clear_chat():
    st.session_state["messages"] = []
    st.session_state["selected_prewritten_prompt"] = None


def sidebar(client_infos: dict[str, ClientInfo], pyproject: dict) -> InferenceSettings:
    st.header(f"💬 Union Chat :gray-badge[v{pyproject['project']['version']}]")
    model = st.selectbox(
        "Select a model",
        list(client_infos),
        key="model_selection",
        on_change=clear_chat,
    )

    # Add max output tokens control
    use_custom_max_tokens = st.toggle(
        "Set custom max output tokens",
        help="Override the model's default max output tokens"
    )
    
    max_output_tokens = None
    if use_custom_max_tokens:
        max_output_tokens = st.number_input(
            "Max output tokens",
            min_value=1,
            max_value=32768,
            value=2048,
            step=1,
            help="Maximum number of tokens to generate in the response"
        )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.01,
        help="Higher values make the output more random, lower values make it more deterministic"
    )

    top_p = st.slider(
        "Top P",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered"
    )

    clear_button = st.button(label="Clear Chat", icon="🗑️")

    if clear_button:
        clear_chat()

    return InferenceSettings(model, max_output_tokens, temperature, top_p)


def on_select(*args, **kwargs):
    st.session_state["selected_prewritten_prompt"] = st.session_state["prewritten-prompt-selection"]


def get_prompt() -> str | None:
    # chat input and pill selector for prewritten prompts
    input = st.chat_input("What is on your mind?")

    # Select from prewritten prompts
    if input:
        selection = None
    elif st.session_state["selected_prewritten_prompt"] is None:
        selection = st.pills(
            "Examples",
            options=[
                "How do I write a file in Python? Be concise.",
                "Write a Haiku about a happy cat",
                "Format the following data into json - name: John Doe, age: 30",
                "What is the capital of Peru?",
                "Translate the following to Spanish: 'Hello, how are you?'",
                "What is 4 * 2 - 1?",
            ],
            key="prewritten-prompt-selection",
            on_change=on_select,
        )
    else:
        selection = st.session_state["selected_prewritten_prompt"]

    # get prompt from input or prewritten prompt
    if input:
        prompt = input
    elif len(st.session_state.messages) == 0:
        prompt = selection
    else:
        prompt = None

    return prompt


def display_messages():
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
        # this is a hack to prevent the message from being displayed twice
        # https://discuss.streamlit.io/t/ghost-double-text-bug/68765/2
        st.empty()


def display_current_prompt_and_response(
    prompt: str,
    client_info: ClientInfo,
    settings: InferenceSettings,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response"):
            stream = client_info.client.chat.completions.create(
                model=client_info.model_id,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                max_tokens=(
                    settings.max_output_tokens
                    if settings.max_output_tokens is not None
                    else client_info.max_tokens
                ),
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
            response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    logger.info("Starting app")

    # load project toml file for the version number
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    # page config and title
    st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:")
    st.title(f"💬 Union Chat")
    st.write(f"**Version:** :gray-badge[v{pyproject['project']['version']}]")
    st.write("A simple UI to chat with Union-hosted LLMs.")

    # load client information
    client_infos, remote_endpoints, headers = load_client_infos()

    # wake up remote endpoints to offset cold start times
    logger.info("Waking up endpoints")
    wake_up_endpoints(remote_endpoints, headers)

    init_session_state()

    with st.sidebar:
        settings = sidebar(client_infos, pyproject)

    # get selected client from the select box
    client_info = client_infos[settings.model]

    with st.container(border=True):
        st.markdown(f"##### 🤖 Current model: **{client_info.model_id}**")
        st.markdown("*:gray[Generated content may be inaccurate or false. Please verify the accuracy of the output.]*")

    prompt = get_prompt()
    display_messages()

    if prompt:
        display_current_prompt_and_response(prompt, client_info, settings)


if __name__ == "__main__":
    main()
