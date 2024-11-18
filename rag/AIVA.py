import requests
import json
import warnings
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext


class AIVA:
    def __init__(self, model="mistral", server_url="http://localhost:11434/api/generate"):
        self._qdrant_url = "https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False,
                                    api_key="zwDHhJD-ZlGfAEYRnNQSTCOKi_eLQFTDa-CbFOS-HMmAmBO00KP89Q")
        self._llm = Ollama(model="mistral", request_timeout=120.0)  # 120 seconds
        self._service_context = ServiceContext.from_defaults(llm=self._llm,
                                                             embed_model="local:sentence-transformers/all-MiniLM-L6-v2")
        self._index = None
        self.model = model
        self.server_url = server_url
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"C:\Users\220425722\Desktop\Python\VAV1\rag\owner_file.txt"]
            )
            documents = reader.load_data()
            # Todo: stop creating the KB if there are no new details in the owner.txt file
            vector_store = QdrantVectorStore(client=self._client, collection_name="va_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def interact_with_llm(self, user_query):
        AgentChatResponse = self._chat_engine.chat(user_query)
        answer = AgentChatResponse.response
        return answer

    def retrieve_from_kb(self, query_text, top_k=3):
        """
        Retrieve relevant documents from Qdrant based on the user query.
        """
        # Use the get_text_embedding function to get embeddings
        query_embedding = self._service_context.embed_model.get_text_embedding(query_text)

        # Perform search in Qdrant
        search_results = self._client.search(
            collection_name="va_db",
            query_vector=query_embedding,
            limit=top_k
        )

        # Extract content from each search result
        context_list = []
        for hit in search_results:
            # Parse the `_node_content` JSON string
            node_content_json = hit.payload.get("_node_content", "{}")
            try:
                node_content = json.loads(node_content_json)
                # Get the actual text content
                text_content = node_content.get("text", "")
                context_list.append(text_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in _node_content: {e}")
                continue

        # Combine all retrieved text for context
        context = " ".join(context_list)
        return context

    # def interact_with_llm(self, user_query):
    #     """
    #     Retrieve relevant context from Qdrant, combine it with the user's query and system prompt,
    #     and send the enriched prompt to the LLM.
    #     """
    #     # Retrieve context from knowledge base
    #     kb_context = self.retrieve_from_kb(user_query)
    #
    #     # Todo: check if the conversation context is being used
    #     # Todo: incorporate general knowledge about older adults. similar to owner.txt
    #     # Combine system prompt, context, and user query
    #     enriched_prompt = f"{self._prompt}\n\nContext:\n{kb_context}\n\nUser: {user_query}" # add user's emotional state here and see how the VA reacts.
    #
    #     payload = {
    #         "model": self.model,
    #         "prompt": enriched_prompt,
    #         "max_tokens": 150  # Adjust max_tokens as needed
    #     }
    #
    #     try:
    #         response = requests.post(
    #             f"{self.server_url}/api/generate",
    #             headers={"Content-Type": "application/json"},
    #             json=payload,
    #             stream=True,
    #             timeout=180  # Set a 3-minute timeout for the response
    #         )
    #         response.raise_for_status()
    #
    #         # Process each line in the streamed response
    #         full_response = ""  # To accumulate the final response text
    #         for line in response.iter_lines():
    #             if line:  # Ensure the line is not empty
    #                 try:
    #                     json_line = json.loads(line)  # Parse JSON line
    #                     part = json_line.get("response", "")
    #                     print("part of the response: ==================== ", part)
    #                     full_response += part
    #                 except json.JSONDecodeError as json_err:
    #                     print(f"JSON decode error: {json_err}")
    #                     continue
    #
    #         return full_response.strip()  # Return the accumulated response
    #
    #     except requests.exceptions.ReadTimeout:
    #         print("Request timed out. Increasing the timeout setting might help.")
    #         return "Sorry, I couldn't process your request due to a timeout. Please try again later."
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error communicating with the local Ollama model: {e}")
    #         return "Sorry, I couldn't process your request at this time."

    @property
    def _prompt(self):
        return """
            You are a warm, friendly, and attentive voice assistant designed to provide companionship and support to socially isolated older adults. Your goal is to engage them in meaningful conversations, offer emotional support, and help them feel connected and valued. Always be patient, empathetic, and encouraging. Your responses should be comforting and cheerful, making them feel like they are talking to a close friend who genuinely cares about their well-being. 

            Balance your conversation with a mix of longer, thoughtful responses and shorter, concise ones to ensure the user has plenty of opportunities to share their thoughts. You can talk about a variety of topics, such as their favorite memories, hobbies, current events, or even guide them through relaxing activities like breathing exercises or listening to music. Be responsive to their needs and emotions, and always prioritize making them feel heard, understood, and appreciated.
            """
