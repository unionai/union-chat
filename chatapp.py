import tomllib
import logging
import os

from dataclasses import dataclass
from models import get_config, PLACEHOLDER_API_KEY, SpecialMessage
from streamlit_local_storage import LocalStorage

from app_utils import wake_up_endpoints

import streamlit as st
from openai import OpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set page config
st.set_page_config(page_title="UnionChat", page_icon=":robot_face:")


local_storage = LocalStorage()


# define types
@dataclass
class ClientInfo:
    client: OpenAI
    model_id: str
    endpoint: str
    public_endpoint: str
    max_tokens: int | None = None
    local: bool = False


@dataclass
class InferenceSettings:
    model: str
    max_output_tokens: int | None
    temperature: float
    top_p: float


@dataclass
class AppConfig:
    show_api_keys: bool
    show_api_keys_message: str
    remote_endpoints: list[str]
    headers: list[dict[str, str]]
    client_infos: dict[str, ClientInfo]
    special_message: SpecialMessage | None

@st.cache_resource
def load_app_config() -> AppConfig:
    config = get_config()

    client_infos = {}
    remote_endpoints, headers = [], []
    for i, model_config in enumerate(config.models):
        endpoint = model_config.get_endpoint(i)
        public_endpoint = model_config.get_public_endpoint(i)

        if model_config.local or model_config.llm_runtime.llm_type == "ollama":
            api_key = "ollama"
        else:
            api_key = os.getenv("UNION_ENDPOINT_SECRET", PLACEHOLDER_API_KEY)
            remote_endpoints.append(endpoint)
            headers.append({"Authorization": f"Bearer {api_key}"})

        client_infos[model_config.display_name] = ClientInfo(
            client=OpenAI(base_url=f"{endpoint}/v1", api_key=api_key),
            model_id=model_config.model_id,
            endpoint=endpoint,
            public_endpoint=public_endpoint,
            max_tokens=model_config.max_tokens,
            local=model_config.local,
        )

    return AppConfig(
        show_api_keys=config.streamlit.show_api_keys,
        show_api_keys_message=config.streamlit.show_api_keys_message,
        remote_endpoints=remote_endpoints,
        headers=headers,
        client_infos=client_infos,
        special_message=config.streamlit.special_message,
    )


def init_session_state():
    # initialize session state
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = None
    if "messages" not in st.session_state:
        clear_chat()
    if "selected_prewritten_prompt" not in st.session_state:
        st.session_state["selected_prewritten_prompt"] = None
    if local_storage.getItem("special_message_shown") is None:
        local_storage.setItem("special_message_shown", 0)


def clear_chat():
    st.session_state["messages"] = []
    st.session_state["selected_prewritten_prompt"] = None


def sidebar(client_infos: dict[str, ClientInfo], pyproject: dict) -> InferenceSettings:
    st.header(f"💬 UnionChat :gray-badge[v{pyproject['project']['version']}]")
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


def _curl_request(app_config: AppConfig, inference_settings: InferenceSettings):
    client_info = app_config.client_infos[inference_settings.model]
    return f"""
curl {client_info.public_endpoint}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -H 'Authorization: Bearer {client_info.client.api_key if app_config.show_api_keys else "xxxxx"}' \\
    -d '{{
        "model": "{client_info.model_id}",
        "messages": [
            {{
                "role": "system",
                "content": "You are a helpful assistant."
            }},
            {{
                "role": "user",
                "content": "Hello!"
            }}
        ]
    }}'
""".strip()


def _python_request(app_config: AppConfig, inference_settings: InferenceSettings):
    client_info = app_config.client_infos[inference_settings.model]
    return f"""
from openai import OpenAI

client = OpenAI(
    base_url = '{client_info.public_endpoint}/v1',
    api_key='{client_info.client.api_key if app_config.show_api_keys else "xxxxx"}'
)

response = client.chat.completions.create(
  model="{client_info.model_id}",
  messages=[
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Who won the world series in 2020?"}},
    {{"role": "assistant", "content": "The LA Dodgers won in 2020."}},
    {{"role": "user", "content": "Where was it played?"}}
  ]
)
print(response.choices[0].message.content)
""".strip()


def _javascript_request(app_config: AppConfig, inference_settings: InferenceSettings):
    client_info = app_config.client_infos[inference_settings.model]
    return f"""
import OpenAI from 'openai'

const openai = new OpenAI({{
  baseURL: '{client_info.public_endpoint}/v1',
  apiKey: '{client_info.client.api_key if app_config.show_api_keys else "xxxxx"}'
}})

const completion = await openai.chat.completions.create({{
  model: '{client_info.model_id}',
  messages: [{{ role: 'user', content: 'Why is the sky blue?' }}],
}})

console.log(completion.choices[0].message.content)
""".strip()


def render_model_info(
    model_info_element,
    app_config: AppConfig,
    inference_settings: InferenceSettings,
    prompt: str | None,
):
    client_info = app_config.client_infos[inference_settings.model]
    with model_info_element.container():
        with st.expander(
            f"##### 🤖 Current model: **{client_info.model_id}**",
            expanded=prompt is None,
        ):
            st.markdown("*:gray[Generated content may be inaccurate or false. Please verify the accuracy of the output.]*")
            col1, col2 = st.columns([.25, 1])

            with col1:
                st.button(
                    "Access via API",
                    on_click=show_api_dialog,
                    kwargs={"app_config": app_config, "inference_settings": inference_settings},
                )
            with col2:
                st.button("Self-deploy", on_click=show_deploy_dialog)


@st.dialog("Access via API", width="large")
def show_api_dialog(app_config: AppConfig, inference_settings: InferenceSettings):
    client_info = app_config.client_infos[inference_settings.model]
    if app_config.show_api_keys_message:
        st.info(app_config.show_api_keys_message, icon="ℹ️")
    else:
        st.write(f"Make requests to the `{client_info.model_id}` model API endpoint:")

    curl_tab, python_tab, js_tab = st.tabs(["Curl", "Python", "JavaScript"])
    with curl_tab:
        st.code(_curl_request(app_config, inference_settings), language="bash")
    with python_tab:
        st.code(_python_request(app_config, inference_settings), language="python")
    with js_tab:
        st.code(_javascript_request(app_config, inference_settings), language="javascript")


@st.dialog("Deploy your own chat UI", width="large")
def show_deploy_dialog():
    st.write("Deploy your own Union Chat UI on Union BYOC or Serverless.")
    st.write("First, clone the repo:")
    st.code("git clone https://github.com/unionai/union-llm-serving\ncd union-llm-serving")
    st.write("Follow the instructions in the [README](https://github.com/unionai/union-llm-serving/blob/main/README.md) to deploy the chat UI.")
    st.write("🤔 Need help? [Contact us](https://www.union.ai/consultation)")


def display_special_message(special_message: SpecialMessage):
    if (
        special_message is not None
        and len(st.session_state.messages) == 2
        and local_storage.getItem("special_message_shown") == 0
    ):
        @st.dialog(special_message.title, width="large")
        def _show_special_message():
            for line in special_message.body:
                st.write(line)

        local_storage.setItem("special_message_shown", 1)
        _show_special_message()


def main():
    logger.info("Starting app")

    # load project toml file for the version number
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    # page title
    st.title(f"💬 UnionChat")
    st.write(f"**Version:** :gray-badge[v{pyproject['project']['version']}]")
    st.write("A simple UI to chat with self-hosted LLMs, powered by [Union](https://www.union.ai).")

    # load client information
    app_config = load_app_config()

    # wake up remote endpoints to offset cold start times
    logger.info("Waking up endpoints")
    wake_up_endpoints(app_config.remote_endpoints, app_config.headers)
    init_session_state()

    if app_config.special_message:
        display_special_message(app_config.special_message)

    with st.sidebar:
        inference_settings = sidebar(app_config.client_infos, pyproject)

    # get selected client from the select box
    client_info = app_config.client_infos[inference_settings.model]

    model_info_element = st.empty()
    prompt = get_prompt()

    render_model_info(model_info_element, app_config, inference_settings, prompt)
    display_messages()

    if prompt:
        display_current_prompt_and_response(prompt, client_info, inference_settings)


if __name__ == "__main__":
    main()
