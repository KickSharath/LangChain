from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Load your LLM
# llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0)
llm = ChatOllama(model="mistral", temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history",  return_messages=True,  output_key="output")

# Functions
def safe_calculator(question: str) -> str:
    try:
        result = eval(question, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return Exception
    
def count_words(text: str) -> str:
    word_count =len(text.split())
    return f"World Count: {word_count}"

def reverse_string(text: str) -> str:
    string_rev = ''.join(reversed(text))
    return f"rev: {string_rev}"

def get_weather(city: str) -> str:
    try:

        return f"weather in {city.title()} is 28 c (mock Data)"
    except Exception as e:
        return f"error; {e}"

def get_crypto_price(coin: str) -> str:
    try:
        prices = {"bitcoin":"60,000 USD", "ethereum": "20,000 USD", "one": "90,00 USD"}
        return f"{coin.title()} price: {prices.get(coin.lower(), 'Not Found')}"
    except Exception as e:
        return f"error: {e}"

# Agents
calculator_tool = Tool(
    name = "Calculator",
    func = safe_calculator,
    description = "Useful for basic math like '72 / 3.6', '5 + 3', etc."
)

count_words_tool = Tool(
    name = "Word Count",
    func = count_words,
    description = "Useful for counting how many words are in a sentence or paragraph." 
)

reverse_string_tool = Tool(
    name = "Reverse String",
    func = reverse_string,
    description = "Useful for Reverse the given string like apple -> elppa"
)

weather_tool = Tool(
    name = "Weather Info",
    func = get_weather,
    description = "Useful for getting Weather in city, it should be a city name like 'Hyderabad' or 'Delhi'"
)

crypto_tool = Tool(
    name = "Crypto Price Checker",
    func = get_crypto_price,
    description = "Useful for getting Crypto price, like Bitcon, Ethereum and One"
)
    

tools = [calculator_tool, count_words_tool, reverse_string_tool, weather_tool, crypto_tool]

agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    memory=memory, # memory past data
    handle_parsing_errors =  True,
    return_intermediate_steps = True,
)

def get_tool_descriptions(tools):
    return "\n".join([f"• {tool.name}: {tool.description}" for tool in tools])

def tool_only_agent_run(user_input: str):
    try:
        response = agent.invoke({"input": user_input}, config={"return_intermediate_steps": True})

        output = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])

        if not intermediate_steps:
            tool_help = get_tool_descriptions(tools)
            return (
                "No tool was used for this query.\n"
                "Available tools:\n" + tool_help
            )

        steps_str = "\n".join([str(step) for step in intermediate_steps])
        return f"Intermediate Steps:\n{steps_str}\n\n💬 Final Answer: {output}"
    except Exception as e:
        return f"⚠️ An error occurred while processing your input:\n{str(e)}"

while True:
    userInput = input("\n KICK: ")

    if userInput.lower() in ["exit"]:
        print('Exiting...')
        break

    response = tool_only_agent_run(userInput)
    print("🧠 Conversation History:\n", memory.buffer)
    print("\nResponse:", response)