import create_data
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import uuid
import sys
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
import os
from langchain import hub
from langchain.vectorstores.chroma import Chroma

# need to load openai api key for gpt3-5 turbo  and embeddings
load_dotenv()
embedding_function = OpenAIEmbeddings()
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.5)


@tool
def read(query_text):
    """
    Get the information by reading from the database and use the retrieved information to generate a response.
    """

    k = 9
    relevance_threshold = 0.75
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    if len(results) == 0 or results[0][1] < relevance_threshold:
        print(f"Unable to find matching results.")
        return None

    context_text = ".".join([doc.page_content for doc, _score in results])
    return "Sources to help respond:  " + context_text


@tool
def add_texts_to_chroma(texts, metadatas=None):
    """
    Saves information by Adding texts to the Chroma database with optional metadatas.
    """
    texts = [texts]
    metadatas = [metadatas]
    # Generate unique identifiers for each text if not provided
    ids = [str(uuid.uuid4()) for _ in texts]
    # Add texts to the database
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return None

class Agent():
    def __init__(self):
        self.llm = chat
        tools = [
            read,
            add_texts_to_chroma,
            self.exit_chat,
        ]
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        self.chat_history = []
        self.Continoue = True

    def launch(self):
        while self.Continoue:
            user_input = input("You: ")
            response = self.agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": self.chat_history,
                }
            )
            self.chat_history.append(response)
    @tool
    def exit_chat(self):
        """
        Leave/Exit the chat.Use it when asked to leave/exit.
        """
        print("Goodbye World!")
        agent.Continoue = False
        return sys.exit(0)


if __name__ == '__main__':
    if not os.path.exists("chroma"):
        print("Creating the database...")
        data = []
        # data = create_data.load_documents("documents") #Optional if you wanna add more data to agent
        db = create_data.save_to_chroma(data)
    else:
        db = Chroma(persist_directory='chroma', embedding_function=embedding_function)
    prompt = hub.pull("hwchase17/react-chat")
    agent = Agent()
    agent.launch()
