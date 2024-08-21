from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import chromadb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.common.config.database import SessionLocal
from app.Books import books_crud
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


client = chromadb.PersistentClient(path="/Users/maltuaijri001/Desktop/Smart-Library/chromadb")
books_collection = client.get_or_create_collection(name="books")

SQLALCHEMY_DATABASE_URL = (
    "postgresql://postgres:48204820Qp$@localhost:5432/smart_library_database"
)
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@tool
def recommend_books(query: str):
    """Search the ChromaDB for relevant book recommendations.
    
    Returns:
    Book recommendations, the recommendations should only be from the retrieved results.
    """
    print('recommend')
    print(query)
    # Perform a similarity search with the query
    results = books_collection.query(
        query_texts=[query],
        n_results=3
    )
    # print(results.get("documents"))
    # Return the titles of the books as recommendations
    return results.get("documents")

# Connect to PostgreSQL (this is a simplified example)

@tool
def summarize_book(title: str):
    """Retrieve a book summary from PostgreSQL based on the title.
    
        Returns:
        if a book is found:
        A coherent summary of the book, Please integrate all the values of the dictionary into the summary and don't correct them.
        if no book is found:
        No summary found for this book.
    """
    print('summarize')
    db = SessionLocal()
    books = books_crud.get_books(db=db, limit=4900)
    for book in books:
        if book.title.lower() == title.lower():
            print(book.title)
            summary = {'Title': book.title, 'Genre': book.genre, 'Author': book.author, 'Description': book.description, 'Published Year': book.published_year, 'Rating': book.average_rating }
            return summary
    print('none found')
    return "No summary found for this book."

# # Define the tools for the agent to use
# @tool
# def chat(query: str):
#     """Chat with the user."""
#     print('chat')
#     return 


tools = [recommend_books, summarize_book]

tool_node = ToolNode(tools)

model = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

template = [
    (
    "You are a smart librarian working in the Smart Library. "
    "If the user wants to chat or greets you then you can respond to them normally. "
    "You only know about the books in the library's database and nothing else. "
    "Do not include any outside knowledge in your responses."
    "Forget every bit of knowledge you have about books that are not in the database"
    "When recommending books, you must first transform the user's query into relevant keywords "
    "The user will enter a query, and you need to understand their intent and convert their prompt into a few relevant keywords. "
    "Try to add more keywords that are similar in meaning to the original keywords. "
    "These keywords will be used as a query to the ChromaDB database for a similarity search. so try to add as many as possible. "
    "After retrieving the recommendations, present them to the user in a friendly way. Do not mention the way you brought them. "
    "Remember to NOT recommend any books other than the books retrieved from the database. "
    "**Always put two spaces before a new line.**"
    ),
    MessagesPlaceholder(variable_name="messages"),
]

prompt = ChatPromptTemplate.from_messages(template)

chain = prompt | model


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # last_content = last_message.content.lower()
    
    # if "recommend" in last_content or "suggest" in last_content:
    #     state['intent'] = "recommend_books"
    #     return "tools"
    
    # if "summary" in last_content or "summarize" in last_content:
    #     state['intent'] = "summarize_book"
    #     return "tools"
    
    # If neither recommend nor summary, assume it's a chat interaction
    state['intent'] = "chat"
 

    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = chain.invoke(messages)

    # Check and print the messages after the model response
    # print(f"Messages after model invocation: {messages}")

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)


# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)


# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')


# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)




def chatbot(query):

    # Use the Runnable
    final_state = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": 42}}
    )

    # print(final_state["messages"][-2].content)
    return final_state["messages"][-1].content

