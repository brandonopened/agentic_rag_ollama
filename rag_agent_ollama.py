from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.base import Embedder
import requests

class OllamaEmbedder(Embedder):
    def __init__(self, model_name="llama3.1:latest"):
        super().__init__()
        self._model_name = model_name
        self._api_base = "http://localhost:11434"
        # Get dimensions by making a test embedding request
        test_embedding = self.get_embedding("test")
        self.dimensions = len(test_embedding)
        
    def get_embedding(self, text):
        response = requests.post(
            f"{self._api_base}/api/embeddings",
            json={
                "model": self._model_name,
                "prompt": text
            }
        )
        response_json = response.json()
        if "embedding" in response_json:
            return response_json["embedding"]
        elif "embeddings" in response_json:
            return response_json["embeddings"]
        else:
            raise KeyError(f"No embeddings found in response: {response_json}")
    
    def get_embedding_and_usage(self, text):
        embedding = self.get_embedding(text)
        # Ollama doesn't provide token usage info
        usage = {"prompt_tokens": 0, "total_tokens": 0}
        return embedding, usage

# Create the embedder
embedder = OllamaEmbedder(model_name="nomic-embed-text")

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=embedder,
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model=Ollama(id="llama3.1:latest"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
