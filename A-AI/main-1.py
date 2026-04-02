import sys
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field

# ==============================
# Shared LLM (used by tools)
# ==============================
expert_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1,
    base_url="http://127.0.0.1:11435"
)

# ==============================
# Tool Input Schema
# ==============================
class QueryInput(BaseModel):
    query: str = Field(description="The user question")

# ==============================
# Tool Definitions
# ==============================
@tool(args_schema=QueryInput)
def math_expert(query: str) -> str:
    """Solve mathematical problems."""
    response = expert_llm.invoke(
        f"You are an expert mathematician. Solve step-by-step:\n{query}"
    )
    return response.content


@tool(args_schema=QueryInput)
def history_expert(query: str) -> str:
    """Answer historical questions with accurate facts."""
    response = expert_llm.invoke(
        f"You are a historian. Provide a clear and factual answer:\n{query}"
    )
    return response.content


@tool(args_schema=QueryInput)
def coding_expert(query: str) -> str:
    """Answer programming and technical questions."""
    response = expert_llm.invoke(
        f"You are a senior software engineer. Help with this:\n{query}"
    )
    return response.content


# ==============================
# Main Application
# ==============================
def main():
    print("\n" + "=" * 70)
    print("LANGCHAIN LOCAL OLLAMA AGENT (LLM-POWERED TOOLS)")
    print("=" * 70 + "\n")

    try:
        # Orchestrator LLM
        print("Initializing Orchestrator...")
        orchestrator_llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0.5,
            base_url="http://127.0.0.1:11435"
        )
        print("✓ Orchestrator ready\n")

        # Bind tools
        tools = [math_expert, history_expert, coding_expert]
        llm_with_tools = orchestrator_llm.bind_tools(tools)

        print("=" * 70)
        print("AGENT READY - Tools:")
        print("• math_expert")
        print("• history_expert")
        print("• coding_expert")
        print("\nType 'exit' to quit")
        print("=" * 70 + "\n")

        # Tool mapping (clean execution)
        tool_map = {
            "math_expert": math_expert,
            "history_expert": history_expert,
            "coding_expert": coding_expert,
        }

        # Main loop
        while True:
            try:
                user_input = input("\n>>> ")

                if user_input.strip().lower() in ["exit", "quit", "terminate"]:
                    print("\n✓ Exiting...")
                    break

                if not user_input.strip():
                    print("(Empty input)")
                    continue

                print("\n[Processing...]")

                # Add system + user message
                messages = [
                    SystemMessage(content="""
You are an intelligent orchestrator.

Routing rules:
- Use math_expert → math problems
- Use history_expert → history questions
- Use coding_expert → programming questions

IMPORTANT:
- Always pass arguments as: {"query": "<user question>"}
- DO NOT generate structured objects like {'type': ...}
- Always follow tool schema strictly
"""),
                    HumanMessage(content=user_input)
                ]

                # First LLM call
                response = llm_with_tools.invoke(messages)

                # Tool loop
                while hasattr(response, "tool_calls") and response.tool_calls:
                    print("\n[Tool Execution]")

                    tool_messages = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # Safe extraction
                        if isinstance(tool_args, dict) and "query" in tool_args:
                            query_arg = tool_args["query"]
                        else:
                            query_arg = str(tool_args)

                        print(f"→ Using {tool_name}: {query_arg}")

                        tool_fn = tool_map.get(tool_name)

                        if tool_fn:
                            result = tool_fn.invoke({"query": query_arg})
                        else:
                            result = "Unknown tool"

                        print("✓ Tool executed")

                        tool_messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"]
                            )
                        )

                    # Append history
                    messages.append(response)
                    messages.extend(tool_messages)

                    # Re-call LLM
                    response = llm_with_tools.invoke(messages)

                # Final output
                print("\n[Final Answer]")
                print(response.content)
                print("\n" + "-" * 70)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as error:
                print(f"\nError: {error}\n")

    except Exception as init_error:
        print(f"\nInitialization Error: {init_error}")
        print("\nMake sure:")
        print("1. Ollama is running")
        print("2. Model exists: ollama pull llama3.2:3b")
        print("3. Correct port: 11435")


if __name__ == "__main__":
    main()