import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts import ChatPromptTemplate

from langchain.chat_models import init_chat_model

#Ask for API key
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

#Initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

#Define prompt template
system_template = "Translate the following from English into {language}"


#Define prompt
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

#Define prompt
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

#Print response
response = model.invoke(prompt)
print(response.content)