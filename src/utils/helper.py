from langchain_core.documents import Document
from typing import Union, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
import base64
from fastapi import UploadFile
from typing import TypeVar

State = TypeVar("State", bound=Dict[str, Any])


def fake_token_counter(messages: Union[list[BaseMessage], BaseMessage]) -> int:
    if isinstance(messages, list):
        return sum(len(message.content.split()) for message in messages)
    return len(messages.content.split())


def convert_list_context_source_to_str(contexts: list[Document]):
    formatted_str = ""
    for i, context in enumerate(contexts):
        formatted_str += f"Document index {i}:\nContent: {context.page_content}\n"
        formatted_str += "----------------------------------------------\n\n"
    return formatted_str


def convert_message(messages):
    list_message = []
    for message in messages:
        if message["type"] == "human":
            list_message.append(HumanMessage(content=message["content"]))
        else:
            list_message.append(AIMessage(content=message["content"]))
    return list_message


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state: State) -> dict:
    error = state.get("error")
    tool_messages = state["build_lesson_plan_response"]
    return {
        "build_lesson_plan_response": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_messages.tool_calls
        ]
    }


def filter_image_messages(messages):
    """
    Filters out messages containing images from a list of message dictionaries.

    Args:
        messages (list): A list of dictionaries, each representing a message with 'role' and 'content' keys.

    Returns:
        list: A new list of dictionaries with messages containing images removed.
    """
    filtered_messages = []

    for message in messages:
        if isinstance(message["content"], list):
            filtered_content = [
                part for part in message["content"] if part.get("type") != "image"
            ]
            if filtered_content:
                print("filtered_content", filtered_content)
                filtered_messages.append(
                    {"role": message["role"], "content": filtered_content[0]["text"]}
                )
        else:
            filtered_messages.append(message)

    return filtered_messages


async def preprocess_messages(query: str, attachs: list[UploadFile]):
    messages: dict[str, list[dict]] = {
        "role": "user",
        "content": [],
    }
    if query:
        messages["content"].append(
            {
                "type": "text",
                "text": query,
            }
        )
    if attachs:
        for attach in attachs:
            if (
                attach.content_type == "image/jpeg"
                or attach.content_type == "image/png"
            ):
                content = await attach.read()
                encoded_string = base64.b64encode(content).decode("utf-8")
                messages["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}",
                        },
                    }
                )
            if attach.content_type == "application/pdf":
                content = await attach.read()
                encoded_string = base64.b64encode(content).decode("utf-8")
                messages["content"].append(
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": "application/pdf",
                        "data": f"{encoded_string}",
                    }
                )
    return messages