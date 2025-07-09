from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", temperature=0)

tools = {
    "Calculator": lambda x: str(eval(x, {"__builtins__": {}})),
    "ReverseText": lambda x: x[::-1],
}

# Custom agent loop
def custom_agent_run(user_input):
    prompt = f"""You are a helpful assistant. You can use tools.

Available tools:
• Calculator: For math like '8 * 5'
• ReverseText: For reversing words like 'apple' → 'elppa'

IMPORTANT: Do NOT guess tool results.
You must wait for the Observation after using a tool.

Answer step by step using this format:
Thought: ...
Action: <Tool Name>
Action Input: <Input for Tool>
Observation: <Tool Output>
...
Final Answer: <Your final answer>

Question: {user_input}
"""

    history = ""
    while True:
        full_prompt = prompt + history
        response = llm.invoke(full_prompt).content.strip()
        print("LLM says:\n", response)

        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        try:
            lines = response.splitlines()
            action_line = [l for l in lines if l.startswith("Action:")][0]
            input_line = [l for l in lines if l.startswith("Action Input:")][0]

            tool_name = action_line.split("Action:")[1].strip()
            tool_input = input_line.split("Action Input:")[1].strip()

            if tool_name in tools:
                observation = tools[tool_name](tool_input)
            else:
                observation = f"Unknown tool: {tool_name}"
        except Exception as e:
            observation = f"Error: {e}"

        history += response + f"\nObservation: {observation}\n"

while True:
    user_input = input("\nKICK: ")
    if user_input.lower() in ['exit']:
        print("Exting...")
        break
    result = custom_agent_run(user_input)
    print("\nFinal Answer:", result)