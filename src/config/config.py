import os
from dotenv import load_dotenv; 
import nest_asyncio, asyncio
from langchain.chat_models import init_chat_model
import openai
from langchain_openai import ChatOpenAI
from langsmith.utils import tracing_is_enabled

nest_asyncio.apply()

load_dotenv(override=True)
print("Attempting to load .env file...")

# API Keys - OpenAIModel will look for OPENAI_API_KEY in environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"OPENAI_API_KEY found in environment. Length: {len(api_key)}, Ends with: ...{api_key[-4:] if len(api_key) > 4 else '****'}")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    print("CRITICAL: OPENAI_API_KEY not found in environment variables. Please ensure it is set in your .env file or system environment.")


os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
print("Tracing activo:", tracing_is_enabled())
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

def configure_asyncio_policy():
    if os.name == 'nt':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("Asyncio policy configured for Windows.")


print("Initializing  OpenAIModels...")
try:

    #model_4_1 = init_chat_model("gpt-4.1")
    model_4_1_mini = init_chat_model("gpt-4.1-mini")
    #model_4_1_nano = init_chat_model("gpt-4.1-nano")
    #model_4_1_turbo = init_chat_model("gpt-4.1-turbo")
    print(" OpenAIModels initialized successfully.")

except Exception as e:
    print(f"CRITICAL ERROR during OpenAIModel initialization: {e}")
    print("This usually means an issue with your OPENAI_API_KEY (not found, invalid, expired, insufficient quota/permissions for the specified models) or the model names.")
    print("Please double-check your .env file for OPENAI_API_KEY and ensure the models (gpt-4.1-nano, gpt-4.1-mini) are accessible with your key.")
    model_nano = None
    model_mini = None

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

model = ChatOpenAI(model="gpt-4.1-mini")   # usa sólo el que necesites