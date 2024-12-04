"""
title: Memodo RAG Pipe-Function
author: Memodo GmbH (mailto:p.oliva@memodo.de)
author_url: https://github.com/memodo
version: 0.1.0
description: This module defines a Pipe class that will be used as a Function in Open WebUI.
requirements: langchain-ollama
"""

from typing import List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import time
import chromadb

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from chromadb.config import Settings
from pydantic import BaseModel

class Pipe:
    class Valves(BaseModel):
        collection_name: str = Field(default="product_management", description="The name of the collection used in the ChromaDB database. The collection name must be between 3 and 63 characters long, start and end with a lowercase letter or a digit, and can only include dots, dashes, and underscores in between.")
        ollama_host: str = Field(default="host.docker.internal")
        ollama_port: str = Field(default="11434")
        chroma_host: str = Field(default="host.docker.internal")
        chroma_port: str = Field(default="8000")
        embeddingModel: str = Field(default="bge-m3:latest")
        chatModel: str = Field(default="llama3.2:3b")
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "memodo_rag_pipe"
        self.name = "Memodo RAG Pipe"
        self.valves = self.Valves()
        self.last_emit_time = 0
        pass

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    async def pipe(self, 
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(
            __event_emitter__, "info", "/initiating Chain", False
        )

        self.model_instance = OllamaLLM(model=self.valves.chatModel, base_url=f"http://{self.valves.ollama_host}:{self.valves.ollama_port}")
        # embeddings = OllamaEmbeddings(model=self.valves.chatModel, base_url=f"http://{self.valves.ollama_host}:{self.valves.ollama_port}")

        client = chromadb.HttpClient(
            self.valves.chroma_host,
            self.valves.chroma_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials="chromadb-test-token",
                chroma_auth_token_transport_header="Authorization"
            )
        )
        self.vector_db_documents = client.get_collection(name=self.valves.collection_name)

        multiquery_template = """
        You are an AI language model assistant.

        Your task is to generate two different versions of the given user question to retrieve relevant documents from a vector database. The two versions should be different from each other and in the same language as the user question.

        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

        Provide these alternative questions separated by newlines. Include the original question as well. Only provide the questions, no other text.

        Original question: {question}
        """
        
        rag_template = """
        You are given a user query, some textual context, rules and steps, all inside xml tags. You have to answer the query based on the context while respecting the rules.

        <context>
        {context}
        </context>

        <rules>
        - If you don't know, just say so.
        - If you are not sure, ask for clarification.
        - Answer in the same language as the user query.
        - If the context appears unreadable or of poor quality, tell the user then answer as best as you can.
        - If the answer is not in the context but you think you know the answer, explain that to the user then answer with your own knowledge.
        - Answer directly and without using xml tags.
        </rules>

        <steps>
        Step 1: Parse Context Information
        Extract and utilize relevant knowledge from the provided context within `<context></context>` XML tags.  
        Step 2: Analyze User Query
        Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.  
        Step 3: Determine Response
        If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.  
        Step 4: Handle Uncertainty
        If the answer is not clear, ask the user for clarification to ensure an accurate response.  
        Step 5: Avoid Context Attribution
        When formulating your response, do not indicate that the information was derived from the context.  
        Step 6: Respond in User's Language
        Maintain consistency by ensuring the response is in the same language as the user's query.  
        Step 7: Provide Response
        Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.  
        </steps>

        <user_query>
        {question}
        </user_query>
        """

        multiquery_prompt = PromptTemplate.from_template(multiquery_template)
        rag_prompt = PromptTemplate.from_template(rag_template)

        chain = (
            { "question": RunnablePassthrough() }
            | multiquery_prompt
            | self.model_instance
            | self.lineListOutputParser
            | {
                "context": lambda x: self.query_db(x),
                "question": lambda _: question
            }
            | rag_prompt
            | self.model_instance
            | StrOutputParser()
        )

        await self.emit_status(
            __event_emitter__, "info", "Starting Memodo RAG Chain", False
        )

        messages = body.get("messages", [])
        
        if messages:
            question = messages[-1]["content"]
            try:
                response = chain.invoke(question)
                # Set assitant message with chain reply
                body["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                await self.emit_status(__event_emitter__, "error", f"Error during Memodo RAG execution: {str(e)}", True)
                return {"error": str(e)}
        else:
            await self.emit_status(__event_emitter__, "error", "No messages found in the request body", True)
            body["messages"].append({"role": "assistant", "content": "No messages found in the request body"})

        await self.emit_status(__event_emitter__, "info", "Memodo RAG Chain complete", True)
        return response
    
    def lineListOutputParser(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return lines  

    def query_db(self, questions):
        results = self.vector_db_documents.query(
            query_texts=questions,
            n_results=2
        )
        return results["documents"]