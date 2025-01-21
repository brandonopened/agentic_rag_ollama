from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo

db_uri = "tmp/lancedb"
# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(table_name="recipes", uri=db_uri, search_type=SearchType.vector),
)
# Load the knowledge base: Comment out after first run
knowledge_base.load(upsert=True)

rag_agent = Agent(
    model=Ollama(model="llama3.1:latest"),  # Changed to use Ollama with llama2
    agent_id="rag-agent",
    knowledge=knowledge_base,
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[rag_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("rag_agent:app", reload=True)