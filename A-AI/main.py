import sys
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def character_counter(text: str) -> str:
    """Counts and returns the number of characters in a given string."""
    return f"Character count: {len(text)}"

@tool
def word_counter(text: str) -> str:
    """Counts and returns the number of words in a given string."""
    return f"Word count: {len(text.split())}"

@tool
def text_analyzer(text: str) -> str:
    """Analyzes text and returns character, word, and line counts."""
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    return f"Lines: {len(lines)}, Words: {words}, Characters: {chars}"

def main():
    print("\n" + "="*70)
    print("LANGCHAIN LOCAL OLLAMA AGENT")
    print("="*70 + "\n")
    
    try:
        # Initialize the local Ollama model
        print("[Step 1] Initializing ChatOllama model...")
        llm = ChatOllama(
            model="llama3.2:3b", 
            temperature=0.7, 
            base_url="http://localhost:11435"
        )
        print("✓ ChatOllama initialized\n")

        # Define the tools available to the agent
        print("[Step 2] Binding tools to LLM...")
        tools = [character_counter, word_counter, text_analyzer]
        llm_with_tools = llm.bind_tools(tools)
        print(f"✓ Bound {len(tools)} tools: character_counter, word_counter, text_analyzer\n")

        print("="*70)
        print("AGENT READY - Available tools:")
        print("  • character_counter: Count characters in text")
        print("  • word_counter: Count words in text")
        print("  • text_analyzer: Analyze text (lines, words, characters)")
        print("\nType 'exit', 'quit', or 'terminate' to end session.")
        print("="*70 + "\n")

        from langchain_core.messages import HumanMessage, ToolMessage

        # Continuous user input loop
        while True:
            try:
                user_input = input("\n[You] >>> ")

                if user_input.strip().lower() in ['exit', 'quit', 'terminate']:
                    print("\n✓ Gracefully shutting down agent.")
                    break

                if not user_input.strip():
                    print("(empty input - please provide a query)")
                    continue

                print("\n[Agent Processing...]")

                messages = [HumanMessage(content=user_input)]

                # Step 1: First LLM call
                response = llm_with_tools.invoke(messages)
                print(f"✓ LLM response received: {response.content}")

                # Step 2: Tool execution loop
                while hasattr(response, "tool_calls") and response.tool_calls:
                    print("\n[Tool Calls Detected]")

                    tool_messages = []

                    for tool_call in response.tool_calls:
                        print(f"→ Tool call: {tool_call}")
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        print(f"→ Calling tool: {tool_name} with args: {tool_args}")

                        # Execute tool
                        if tool_name == "character_counter":
                            result = character_counter.invoke(tool_args)
                        elif tool_name == "word_counter":
                            result = word_counter.invoke(tool_args)
                        elif tool_name == "text_analyzer":
                            result = text_analyzer.invoke(tool_args)
                        else:
                            result = "Unknown tool"

                        print(f"✓ Tool result: {result}")

                        # IMPORTANT: Send result back to LLM
                        tool_messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"]
                            )
                        )

                    # Add assistant message + tool results
                    messages.append(response)
                    messages.extend(tool_messages)

                    # Call LLM again with tool results
                    response = llm_with_tools.invoke(messages)

                # Final response
                print(f"\n[Agent Response]\n{response.content}")
                print("\n" + "-" * 70)

            except KeyboardInterrupt:
                print("\n\n✓ Interrupted by user.")
                break
            except Exception as error:
                print(f"\n✗ Error during execution: {error}")
                print("Continuing...\n")

    except Exception as init_error:
        print(f"\n✗ Initialization Error: {init_error}\n")
        print("TROUBLESHOOTING STEPS:")
        print("1. Start Ollama server in another terminal:")
        print("   ollama serve")
        print("2. Verify model is installed:")
        print("   ollama pull llama3.2:3b")
        print("3. Test connection:")
        print("   curl http://localhost:11435/api/tags")
        print("4. Check firewall - ensure port 11435 is accessible")

if __name__ == "__main__":
    main()