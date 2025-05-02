import tomllib
from dataclasses import dataclass
from models import get_config

import httpx
import streamlit as st
from openai import OpenAI


with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:")
st.title(f"💬 Union Chat")
st.write(f"**Version:** :gray-badge[v{pyproject['project']['version']}]")


@dataclass
class ClientInfo:
    client: OpenAI
    model_id: str
    endpoint: str
    max_tokens: int | None = None

@st.cache_resource
def load_client_infos() -> dict[str, ClientInfo]:
    config = get_config()

    client_infos = {}
    for i, model_config in enumerate(config.models):
        endpoint = model_config.get_endpoint(i)
        api_key = "ollama" if model_config.local else "ABC"
        client_infos[model_config.display_name] = ClientInfo(
            client=OpenAI(base_url=f"{endpoint}/v1", api_key=api_key),
            model_id=model_config.model_id,
            endpoint=endpoint,
            max_tokens=model_config.max_tokens,
        )

    return client_infos


client_infos = load_client_infos()


def clear_chat():
    st.session_state["messages"] = []
    st.session_state["selected_prewritten_prompt"] = None


if "current_model" not in st.session_state:
    st.session_state["current_model"] = None
if "messages" not in st.session_state:
    clear_chat()
if "selected_prewritten_prompt" not in st.session_state:
    st.session_state["selected_prewritten_prompt"] = None

with st.sidebar:
    st.header(f"💬 Union Chat :gray-badge[v{pyproject['project']['version']}]")
    select_box = st.selectbox(
        "Select a model",
        list(client_infos),
        key="model_selection",
        on_change=clear_chat,
    )

    clear_button = st.button(label="Clear Chat", icon="🗑️")

    if clear_button:
        clear_chat()


client_info = client_infos[select_box]

with st.container(border=True):
    st.markdown(f"##### 🤖 Current model: **{client_info.model_id}**")
    st.markdown("*:gray[Generated content may be inaccurate or false.]*")

placeholder = st.empty()

def on_select(*args, **kwargs):
    st.session_state["selected_prewritten_prompt"] = st.session_state["prewritten-prompt-selection"]

# Select from prewritten prompts
if not st.session_state["selected_prewritten_prompt"]:
    selection = placeholder.pills(
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


# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input = st.chat_input("What is on your mind?")


if input:
    prompt = input
else:
    prompt = selection

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        stream = client_info.client.chat.completions.create(
            model=client_info.model_id,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            max_tokens=client_info.max_tokens,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
