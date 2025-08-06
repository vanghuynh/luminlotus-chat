from typing import TypedDict, Optional, List
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing import Sequence, Annotated
from langchain_core.messages import RemoveMessage
from langchain_core.documents import Document
from src.utils.logger import logger
from src.config.llm import get_llm
from .prompt import system_prompt, template_prompt
from .tools import check_active_coupons, extract_query_product,predict_size_model, check_order_status, extract_information_product


class State(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    selected_ids: Optional[List[str]]
    selected_documents: Optional[List[Document]]


from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=get_llm("gemini-2.0-flash"),  # or "gpt-4o"
    tools=[extract_query_product,predict_size_model, check_order_status, extract_information_product, check_active_coupons],  
    prompt=template_prompt, 
    checkpointer=checkpointer,
)
