import os
from dataclasses import dataclass
from ruamel.yaml import YAML
from pathlib import Path

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:")
st.title("Union Serving LLMs")


@dataclass
class ClientInfo:
    client: OpenAI
    model_id: str


@st.cache_resource
def load_client_infos():
    # Load client config from file
    config_file = os.getenv("CONFIG_FILE")
    if config_file is None:
        msg = "CONFIG_FILE must be set"
        raise RuntimeError(msg)

    with open(config_file) as f:
        yaml = YAML(typ="safe")
        config = yaml.load(f)

    client_infos = {}
    for model_config in config["models"]:
        if "base_url" in model_config:
            endpoint = model_config["base_url"]
        elif "base_url_env_var" in model_config:
            env_var = model_config["base_url_env_var"]
            endpoint = os.getenv(env_var, "")
            if endpoint == "":
                msg = "base_url_env_var is not set"
                raise RuntimeError(msg)
        else:
            msg = "base_url or base_url_end_var must be set"
            raise RuntimeError(msg)

        client_infos[model_config["name"]] = ClientInfo(
            client=OpenAI(base_url=f"{endpoint}/v1", api_key="ABC"),
            model_id=model_config["model_id"],
        )

    return client_infos


client_infos = load_client_infos()


def clear_chat():
    st.session_state["messages"] = []


if "messages" not in st.session_state:
    clear_chat()

with st.sidebar:
    select_box = st.selectbox(
        "Model Selection",
        list(client_infos),
        key="model_selection",
        on_change=clear_chat,
    )

    clear_button = st.button(label="Clear Chat", icon="🗑️")

    if clear_button:
        clear_chat()


st.markdown(f"#### Chatting with {select_box}")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

client_info = client_infos[select_box]

if prompt := st.chat_input("What is on your mind?"):
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
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
