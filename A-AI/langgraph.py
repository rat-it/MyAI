import sys
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Define the Application State
# The 'add_messages' reducer ensures that new messages are appended to the 
# list rather than overwriting the entire conversation history.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str

# 2. Define the Routing Schema for the Supervisor
# We use Pydantic to force the LLM to output a specific structured choice.
class Route(BaseModel):
    next_node: Literal

def main():
    print("\n" + "="*70)
    print("LANGGRAPH MULTI-AGENT SUPERVISOR (OLLAMA)")
    print("="*70 + "\n")

    try:
        # Initialize the local Ollama LLM
        # Low temperature is used to keep the expert responses highly factual.
        llm = ChatOllama(model="llama3.2:3b", temperature=0.1, base_url="http://localhost:11434")
        
        # 3. Define the Nodes (The Agents)
        def supervisor_node(state: AgentState):
            """The supervisor analyzes the conversation and routes to the correct expert."""
            system_prompt = (
                "You are a supervisor managing a 'math_expert', a 'history_expert', and a 'coding_expert'. "
                "Analyze the user's message and decide who should answer. "
                "If the user wants to exit, says hello, or the task is complete, return 'FINISH'."
            )
            
            # Bind the LLM to output our exact Pydantic schema
            router_llm = llm.with_structured_output(Route)
            
            messages = [{"role": "system", "content": system_prompt}] + state["messages"]
            response = router_llm.invoke(messages)
            
            print(f"\n Routing task to -> {response.next_node}")
            return {"next_node": response.next_node}

        def math_node(state: AgentState):
            """Expert node for mathematics."""
            print("[Math Expert] Analyzing formula and computing...")
            messages = + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [AIMessage(content=f"🧮 [Math]: {response.content}")]}

        def history_node(state: AgentState):
            """Expert node for history."""
            print("[History Expert] Retrieving historical records...")
            messages = + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [AIMessage(content=f"🏛️ [History]: {response.content}")]}

        def coding_node(state: AgentState):
            """Expert node for programming."""
            print("[Coding Expert] Compiling technical response...")
            messages = + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [AIMessage(content=f"💻 [Code]: {response.content}")]}

        # 4. Define the Conditional Routing Function
        def route_from_supervisor(state: AgentState):
            """Reads the state updated by the supervisor and directs the graph."""
            if state["next_node"] == "FINISH":
                return END
            return state["next_node"]

        # 5. Build the LangGraph Workflow
        builder = StateGraph(AgentState)

        # Add all the nodes to the graph
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("math_expert", math_node)
        builder.add_node("history_expert", history_node)
        builder.add_node("coding_expert", coding_node)

        # Define the execution flow
        builder.add_edge(START, "supervisor") # The workflow always starts at the supervisor
        
        # The supervisor dynamically points to an expert or ends the workflow using conditional edges
        builder.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "math_expert": "math_expert",
                "history_expert": "history_expert",
                "coding_expert": "coding_expert",
                END: END
            }
        )

        # Once an expert finishes generating their response, the workflow concludes
        builder.add_edge("math_expert", END)
        builder.add_edge("history_expert", END)
        builder.add_edge("coding_expert", END)

        # Compile the graph into an executable application
        graph = builder.compile()
        print("✓ Multi-Agent LangGraph compiled successfully.\n")

        # 6. Continuous User Interaction Loop
        while True:
            try:
                user_input = input("\n[User] >>> ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    print("\nShutting down multi-agent system.")
                    break
                if not user_input.strip():
                    continue

                # Invoke the graph with the initial state
                result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
                
                # Print the final message added to the state by the selected expert
                print("\n" + "="*70)
                print(result["messages"][-1].content)
                print("="*70)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting.")
                break
            except Exception as loop_err:
                print(f"\nError during graph execution: {loop_err}")

    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Ensure Ollama is running and 'llama3.2:3b' is pulled.")
        print("You may need to run: pip install pydantic langgraph langchain-ollama")

if __name__ == "__main__":
    main()