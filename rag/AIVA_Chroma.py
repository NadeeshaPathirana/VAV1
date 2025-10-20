import time

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from bs4 import BeautifulSoup
import requests
from googleapiclient.discovery import build
from typing import List


from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding

class LocalHFEmbedding(BaseEmbedding):
    model: SentenceTransformer  # declare as field

    # Use a classmethod constructor instead of __init__ for offline model
    @classmethod
    def from_local_path(cls, model_path: str):
        model = SentenceTransformer(model_path)
        return cls(model=model)

    # Required sync embedding methods
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def _get_query_embedding(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    # async version
    async def _aget_query_embedding(self, texts: List[str]) -> List[List[float]]:
        return self._get_query_embedding(texts)

# 3rd version of AIVA. Use Chroma DB instead of Qdrant
class AIVA_Chroma:
    def __init__(self, model="mistral"):
        self._chroma_client = chromadb.PersistentClient(path="./chroma_db")
        if self._chroma_client is None:
            raise RuntimeError("Failed to initialize ChromaDB PersistentClient.")
        self._llm = Ollama(model="mistral", request_timeout=120.0)  # 120 seconds

        embed_model = LocalHFEmbedding(
            model=SentenceTransformer(
                r"C:\Users\220425722\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
            )
        )  # path to downloaded model

        self._service_context = ServiceContext.from_defaults(
            llm=self._llm,  # your local LLM object
            embed_model=embed_model
        ) # a lightweight Sentence Transformer model
        self._index = None
        self.model = model
        # self._api_key = 'AIzaSyAB_yU07EvwEc2D0pK8hJhoxjQZPwFUHxc'
        self._api_key = 'AIzaSyCWBlzpNEEgOkb3GsYdB3SDIOvUmr_h1ig'
        self._cse_id = '5086429ea12f641aa'
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=self._memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"C:\Users\220425722\Desktop\Python\VAV1\rag\owner_file.txt"]
            )
            documents = reader.load_data()

            # Ensure documents are not empty
            if not documents:
                raise ValueError("No documents found. Ensure the file exists and is not empty.")

            # Ensure ChromaDB client is initialized
            if self._chroma_client is None:
                raise RuntimeError("Failed to initialize ChromaDB PersistentClient.")

            # Ensure collection is created
            collection = self._chroma_client.get_or_create_collection("va_db")
            if collection is None:
                raise RuntimeError("Failed to create or retrieve ChromaDB collection.")

            vector_store = ChromaVectorStore(chroma_collection=collection, collection_name="va_db")

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create the index
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )

            # save index
            self._index.storage_context.persist(persist_dir="storage") # TODO: write code to reuse the stored indices

            print("Knowledgebase created successfully using ChromaDB!")

        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            self._index = None

    # Initialize Google Custom Search API
    def google_search(self, query, api_key, cse_id, num_results=3):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return res.get("items", [])

    # Scrape content from a URL using BeautifulSoup
    def scrape_page(self, url):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")  # Extract text from paragraphs
            return ' '.join([p.get_text() for p in paragraphs if p.get_text().strip()])
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def interact_with_llm(self, user_query, emotion):
        start_time = time.time()
        # search_results = self.google_search(user_query, self._api_key, self._cse_id, num_results=3)

        web_data = ""
        # for result in search_results:
        #     url = result.get('link')
        #     print(f"Scraping {url}...")
        #     content = self.scrape_page(url)
        #     if content:
        #         web_data += content + "\n\n"  # Add the content of each page

        # print(f"web_data {web_data}")
        # Combine web data with the user query for context
        # combined_query = user_query + "\n\n" + "Use the following information retrieved from the internet when you answer the query:\n" + web_data

        AgentChatResponse = self._chat_engine.chat(user_query)
        answer = AgentChatResponse.response
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        print(f"LLM Interaction Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
        return answer

    @property
    def _prompt(self):
        return """
            You are a warm, friendly, and attentive voice assistant designed to provide companionship and support to socially isolated older adults. Your goal is to engage them in meaningful conversations, offer emotional support, and help them feel connected and valued. Always be patient, empathetic, non-intrusive, and encouraging. Your responses should be comforting and cheerful, making them feel like they are talking to a close friend who genuinely cares about their well-being. 

            Give short answers. You can pick a topic, such as their favorite memories, hobbies, current events, or even guide them through relaxing activities like breathing exercises or listening to music. Be responsive to their needs and emotions, and always prioritize making them feel heard, understood, and appreciated. Give responses in spoken language.
            """
