import time

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.llms import ChatMessage
from bs4 import BeautifulSoup
import requests
from llama_index.core import Document
from typing import List
import os
import json

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
        self._llm = Ollama(model="mistral", request_timeout=240.0)  # 240 seconds - increased to solve the timeout issue

        embed_model = LocalHFEmbedding(
            model=SentenceTransformer(
                r"C:\Users\220425722\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
            )
        )  # path to downloaded model

        self._service_context = ServiceContext.from_defaults(
            llm=self._llm,  # your local LLM object
            embed_model=embed_model
        )  # a lightweight Sentence Transformer model
        self._index = None
        self.model = model
        # self._api_key = 'AIzaSyAB_yU07EvwEc2D0pK8hJhoxjQZPwFUHxc'
        self._api_key = 'AIzaSyCWBlzpNEEgOkb3GsYdB3SDIOvUmr_h1ig'
        self._cse_id = '5086429ea12f641aa'
        self._big5_score = {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5} # assuming 0.5 is the default value
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=self._memory,
            system_prompt=self._prompt,
            similarity_top_k=2
        )

    def get_custom_prompt(self, emotion: str):
        custom_prompt = self._prompt
        return custom_prompt

    def load_profile_files(self, profile_dir: str):
        """
        Load and flatten all JSON and TXT files from a directory into LlamaIndex Documents.
        """
        documents = []

        for file_name in os.listdir(profile_dir):
            file_path = os.path.join(profile_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            # --- JSON FILES ---
            if file_name.lower().endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    def flatten_json(d, prefix=""):
                        items = []
                        for k, v in d.items():
                            if isinstance(v, dict):
                                items += flatten_json(v, f"{prefix}{k}.")
                            elif isinstance(v, list):
                                for i, val in enumerate(v):
                                    items.append(f"{prefix}{k}[{i}]: {val}")
                            else:
                                items.append(f"{prefix}{k}: {v}")
                        return items

                    flattened_texts = flatten_json(data)
                    for text in flattened_texts:
                        documents.append(Document(
                            text=text,
                            metadata={"source": file_name, "type": "profile_json"}
                        ))
                        if file_name == "user_profile.json":
                            self._create_personality_score_map(text)

                except Exception as e:
                    print(f"[WARN] Failed to load JSON {file_name}: {e}")

            else:
                print("Profile is not a json file")

        print(f"[INFO] Loaded {len(documents)} profile documents from {profile_dir}")
        return documents

    def _create_kb(self):
        try:
            profile_dir = r"C:\Users\220425722\Desktop\Python\VAV1\rag\profile"

            # Load and prepare documents
            documents = self.load_profile_files(profile_dir)

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
            self._index.storage_context.persist(persist_dir="storage")  # TODO: write code to reuse the stored indices

            print("Knowledgebase created successfully using ChromaDB!")

        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            self._index = None

    # def interact_with_llm(self, user_query, emotion=None, custom_prompt=None):
    #     start_time = time.time()
    #
    #     dynamic_prompt = custom_prompt if custom_prompt else self._prompt
    #     self._chat_engine = self._index.as_chat_engine(
    #         chat_mode="context",
    #         memory=self._memory,
    #         system_prompt=dynamic_prompt,
    #         similarity_top_k=2
    #     )
    #
    #     AgentChatResponse = self._chat_engine.chat(user_query)
    #     answer = AgentChatResponse.response
    #     end_time = time.time()  # End time measurement
    #     execution_time = end_time - start_time
    #     print(f"LLM Interaction Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
    #     return answer

    def interact_with_llm(self, user_query, emotion=None, custom_prompt=None):
        start_time = time.time()
        # custom_prompt = self.get_custom_prompt(emotion) # todo: test both scenarios
        #
        # dynamic_prompt = custom_prompt if custom_prompt else self._prompt
        #
        # self._chat_engine = self._index.as_chat_engine(
        #     chat_mode="context",
        #     memory=self._memory,
        #     system_prompt=dynamic_prompt,
        #     similarity_top_k=2,
        # )

        try:
            memory_text = " ".join([m.content for m in self._memory.get_all()])  # get stored messages
            memory_token_count = len(memory_text.split())  # rough estimate (1 token ≈ 1 word)

            if memory_token_count > 1200:  # close to your 1500 limit
                print(f"[INFO] Memory near token limit ({memory_token_count} tokens). Summarizing...")
                summary_prompt = (
                    "Summarize the following conversation briefly in 3-4 sentences, keeping emotional tone:"
                )
                # Summarize memory using LLM (lightweight)
                summary_response = self._llm.complete(summary_prompt + "\n\n" + memory_text[:5000])
                summary = summary_response.text if hasattr(summary_response, "text") else str(summary_response) #todo: might need to save in the DB

                # Reset memory and insert summarized version
                self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
                self._memory.put(ChatMessage(role="system", content=f"Conversation summary: {summary}"))
                print("[INFO] Memory summarized successfully.")

        except Exception as e:
            print(f"[WARN] Memory summarization failed: {e}")

        try:
            AgentChatResponse = self._chat_engine.chat("user said this: " + "\'" + user_query + "\'" + "User seems to be feeling: " + emotion + "reply the user with the same emotional tone except in anger "+emotion) #todo: call a method with emotional reply logic if/else
            answer = AgentChatResponse.response
        except Exception as e:
            print(f"[ERROR] Chat failed due to {e}. Resetting memory to recover.")
            self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            answer = "I'm sorry, I had a little trouble remembering everything just now — but I’m back!"

        end_time = time.time()
        print(f"LLM Interaction Execution Time: {end_time - start_time:.2f} seconds")
        return answer

    def get_personality_vs_com_style_query(self):
        output = ""
        if self._big5_score["openness"] >= 0.5:
            output = "this person is like this.... and would like to be communicated like this.. in this emotional tone, and in this... interaction style. When you are replying to the older adult, apart from their personality traits, consider their current emotion as well. You will get to know their current emotion with each conversation round of utterance."
        elif self._big5_score["openness"] < 0.5:
            output = ""

        if self._big5_score["conscientiousness"] >= 0.5:
            output = ""
        elif self._big5_score["conscientiousness"] < 0.5:
            output = ""

        if self._big5_score["extraversion"] >= 0.5:
            output = ""
        elif self._big5_score["extraversion"] < 0.5:
            output = ""

        if self._big5_score["agreeableness"] >= 0.5:
            output = ""
        elif self._big5_score["agreeableness"] < 0.5:
            output = ""

        if self._big5_score["neuroticism"] >= 0.5:
            output = ""
        elif self._big5_score["neuroticism"] < 0.5:
            output = ""

        return output
    @property
    def _prompt(self): #todo: needs to be changed to accommodate the first interaction
        return """
            You are a warm, friendly, and attentive voice assistant designed to provide companionship and support to socially isolated older adults. Your goal is to engage them in meaningful conversations, offer emotional support, and help them feel connected and valued. Always be patient, empathetic, non-intrusive, and encouraging. Your responses should be comforting and cheerful, making them feel like they are talking to a close friend who genuinely cares about their well-being. 

            Give short answers, not more than 30 words. You can pick a topic, such as their favorite memories, hobbies, current events, or even guide them through relaxing activities like breathing exercises or listening to music. Be responsive to their needs and emotions, and always prioritize making them feel heard, understood, and appreciated. Give responses in spoken language.
            
            Do not start the conversation with all the information you know about the user. Gradually go in to details about the user. When required, use the information in the user profiles. But do not over use and be repetitive.
            If the user starts a topic, continue the conversation along that direction without changing the conversation topic on your own.
            
            Note that women are more agreeable and expressive while communicating compared to men, and men are more precise in their communication. 
            """ + self.get_personality_vs_com_style_query()

    def _create_personality_score_map(self, text):
        if text.split(":")[0].strip() == 'personality_traits.openness_score':
            self._big5_score["openness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.conscientiousness_score':
            self._big5_score["conscientiousness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.extraversion_score':
            self._big5_score["extraversion"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.agreeableness_score':
            self._big5_score["agreeableness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.neuroticism_score':
            self._big5_score["neuroticism"] = float(text.split(":")[1].strip())
