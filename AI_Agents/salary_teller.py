from dotenv import load_dotenv

_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

tool = TavilySearchResults(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)  # Use the checkpointer passed as a parameter
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

if __name__ == "__main__":

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    model = ChatOpenAI(model="gpt-4o-2024-08-06")
    tool = TavilySearchResults(max_results=2)

    # Use SqliteSaver as a context manager to keep it open during graph usage
    with SqliteSaver.from_conn_string(":memory:") as memory:
        # Instantiate the agent with the open checkpointer
        abot = Agent(model, [tool], checkpointer=memory, system=prompt)

        messages = [HumanMessage(content="What is the weather in sf?")]

        # Now use the graph stream while the SqliteSaver connection is still open
        for event in abot.graph.stream({"messages": messages}, {"configurable": {"thread_id": "1"}}):
            for v in event.values():
                print(v['messages'])
                
        print('#####################################################################################')        
        result = abot.graph.invoke({"messages": messages}, {"configurable": {"thread_id": "1"}})
        print(result['messages'][-1].content)        