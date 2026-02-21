import os
import sys
import yaml
import glob
import time
import queue
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import logging

# Original imports removed for debugging
# from langchain_core.vectorstores import InMemoryVectorStore
# ...
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

import langchain
langchain.verbose = False

print("DEBUG: chatbot.py imports starting...", flush=True)

from langchain_core.vectorstores import InMemoryVectorStore
print("DEBUG: Imported InMemoryVectorStore", flush=True)
from langchain_community.embeddings import HuggingFaceEmbeddings
print("DEBUG: Imported HuggingFaceEmbeddings", flush=True)
from langchain_groq import ChatGroq
print("DEBUG: Imported ChatGroq", flush=True)

# try:
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     print("DEBUG: Imported ChatGoogleGenerativeAI", flush=True)
# except ImportError as e:
ChatGoogleGenerativeAI = None
#     with open("debug_import_error.txt", "w") as f:
#         f.write(f"ImportError: {e}\n")
#     print(f"DEBUG: ChatGoogleGenerativeAI not found: {e}", flush=True)

# OpenAI support removed per user request
# from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
print("DEBUG: Imported ConversationalRetrievalChain, RetrievalQA", flush=True)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader,
    CSVLoader
)
print("DEBUG: Imported document loaders", flush=True)
from langchain.prompts import PromptTemplate
print("DEBUG: chatbot.py imports finished", flush=True)

class ManualRAG:
    """
    A lightweight RAG implementation to bypass LangChain Chain issues on Python 3.14.
    """
    def __init__(self, llm, retriever, prompt):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        
    def invoke(self, inputs: dict, config=None):
        query = inputs.get("query") or inputs.get("question")
        if not query:
            return {"result": "No query provided."}
            
        # Prepare query for retrieval
        search_query = query
        chat_history = inputs.get("chat_history", [])
        
        # Condense question if history exists
        if chat_history and self.llm:
            try:
                # Convert list of messages to string if needed
                if isinstance(chat_history, list):
                     chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
                else:
                     chat_history_str = str(chat_history)
                
                condense_prompt = f"""Given the following conversation and a follow-up user input, rephrase the follow-up input to be a standalone question.
Chat History:
{chat_history_str}
Follow Up Input: {query}
Standalone Question:"""
                
                condensed = self.llm.invoke(condense_prompt)
                search_query = condensed.content if hasattr(condensed, 'content') else str(condensed)
                print(f"DEBUG: Condensed query: '{query}' -> '{search_query}'", flush=True)
            except Exception as e:
                print(f"DEBUG: Condensation failed: {e}", flush=True)

        # Retrieve documents using strict search query
        docs = self.retriever.get_relevant_documents(search_query)
        docs_text = "\n\n".join([d.page_content for d in docs])
        
        # Format prompt
        # Ensure prompt template uses 'context', 'question', and 'chat_history'
        chat_history = inputs.get("chat_history", [])
        # Convert list of messages to string if needed, or pass as is if prompt expects it
        if isinstance(chat_history, list):
             chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
        else:
             chat_history_str = str(chat_history)
             
        final_prompt = self.prompt.format(
            context=docs_text, 
            question=query,
            chat_history=chat_history_str
        )
        
        # Generate answer
        # Supports ChatModels (invoke returns AIMessage) and LLMs (returns str)
        response = self.llm.invoke(final_prompt)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "result": answer,
            "answer": answer, # For compatibility
            "source_documents": docs
        }

class UniMateChatbot:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the UniMate Chatbot with Groq support"""
        print("DEBUG: UniMateChatbot __init__ called")
        
        # Determine config path absolute or relative
        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            # Assume relative to project root (parent of backend)
            self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize paths relative to project root
        project_root = Path(os.path.dirname(os.path.dirname(__file__)))
        
        with open("debug_loading.log", "a") as f:
            f.write(f"DEBUG: Project Root: {project_root}\n")
        
        # Handle data_dirs list from config
        self.data_dirs = []
        if 'data_dirs' in self.config['paths']:
            for d in self.config['paths']['data_dirs']:
                path = project_root / d
                with open("debug_loading.log", "a") as f:
                    f.write(f"DEBUG: Configured data dir: {d} -> Resolved: {path}\n")
                
                # path.mkdir(parents=True, exist_ok=True) # Dont create if investigating
                if not path.exists():
                     with open("debug_loading.log", "a") as f:
                        f.write(f"DEBUG: PATH DOES NOT EXIST: {path}\n")
                self.data_dirs.append(path)
        else:
            # Fallback for old config
            d_dir = project_root / self.config['paths'].get('data_dir', './data')
            d_dir.mkdir(parents=True, exist_ok=True)
            self.data_dirs.append(d_dir)
            
        self.vector_store_path = project_root / self.config['paths']['vector_store']
        
        # Track costs and performance
        self.query_count = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        # Initialize models
        self.embedding_model = self._init_embedding_model()
        self.google_llm = None
        self.groq_llm = None
        self.llm = self._init_llm()
        
        # Initialize components
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.processed_files = set()
        
        # Auto-load if enabled
        if self.config.get('auto_load', True):
            self.auto_load_documents()
    
    def _init_embedding_model(self):
        """Initialize FREE local embedding model"""
        model_name = self.config['embeddings']['model']
        
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _init_llm(self):
        """Initialize LLM based on provider choice"""
        # Support new and old config structure
        if 'llm' in self.config:
            provider = self.config['llm']['provider'].lower()
            model_conf = self.config['llm']
        else:
            provider = self.config.get('llm_provider', 'groq').lower()
            model_conf = self.config.get('groq', {})

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("Groq API key not found in .env file")
                return None
            
            # Get model name from new or old config
            model_name = model_conf.get('model_name') or model_conf.get('model') or "llama-3.1-8b-instant"
            
            groq_models = {
                "llama-3.3-70b-versatile": "Llama 3.3 70B",
                "llama-3.1-8b-instant": "Llama 3.1 8B",
                "llama3-70b-8192": "Llama 3.1 70B",
                "mixtral-8x7b-32768": "Mixtral 8x7B",
                "gemma2-9b-it": "Gemma 2 9B",
                "llama3-8b-8192": "Llama 3.1 8B"
            }
            
            # if model_name not in groq_models:
            #     logger.warning(f"Model {model_name} not recognized. Using Llama 3.3 70B")
            #     model_name = "llama-3.3-70b-versatile"
            
            logger.info(f"Using Groq: {groq_models.get(model_name, model_name)}")
            
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=model_conf.get('temperature', 0.1),
                max_tokens=model_conf.get('max_tokens', 1024),
                streaming=True
            )
        
        # OpenAI block removed

        elif provider == "google":
            # Determine Google model config based on new/old structure
            if 'llm' in self.config and self.config['llm']['provider'].lower() == 'google':
                google_model_conf = self.config['llm']
            else:
                google_model_conf = self.config.get('google', {})

            print(f"DEBUG: Initializing Google LLM with model: {google_model_conf.get('model_name') or google_model_conf.get('model')}")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("Google API key not found. Falling back to Groq...")
                return self._fallback_to_groq()
            
            if not ChatGoogleGenerativeAI:
                logger.warning("langchain-google-genai not installed. Falling back to Groq...")
                return self._fallback_to_groq()

            self.google_llm = ChatGoogleGenerativeAI(
                model=google_model_conf.get('model_name') or google_model_conf.get('model'),
                temperature=google_model_conf.get('temperature', 0.1),
                max_output_tokens=google_model_conf.get('max_tokens', 1024),
                google_api_key=api_key,
                convert_system_message_to_human=True,
                streaming=True
            )
            return self.google_llm
        
        elif provider == "hybrid":
            print("DEBUG: Initializing Hybrid Mode (Groq + Google)")
            
            # Init Groq for Retrieval
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                # Use new config logic
                if 'llm' in self.config:
                    g_model = self.config['llm'].get('model_name') or self.config['llm'].get('model')
                    g_temp = self.config['llm'].get('temperature', 0.1)
                    g_max_tokens = self.config['llm'].get('max_tokens', 1024)
                else:
                    g_model = self.config['groq'].get('model')
                    g_temp = self.config['groq'].get('temperature', 0.1)
                    g_max_tokens = self.config['groq'].get('max_tokens', 1024)
                    
                self.groq_llm = ChatGroq(
                    groq_api_key=groq_key,
                    model_name=g_model,
                    temperature=g_temp,
                    max_tokens=g_max_tokens,
                    streaming=True
                )
            
            # Init Google for Polishing
            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key and ChatGoogleGenerativeAI:
                # Determine Google model config based on new/old structure
                if 'llm' in self.config and self.config['llm']['provider'].lower() == 'hybrid':
                    google_model_conf = self.config['llm'] # Hybrid uses the same 'llm' block for both
                else:
                    google_model_conf = self.config.get('google', {})

                self.google_llm = ChatGoogleGenerativeAI(
                model=google_model_conf.get('model_name') or google_model_conf.get('model'),
                temperature=google_model_conf.get('temperature', 0.1),
                max_output_tokens=google_model_conf.get('max_tokens', 1024),
                google_api_key=google_key,
                convert_system_message_to_human=True,
                streaming=True
            )
            
            if not self.groq_llm or not self.google_llm:
                logger.warning("Hybrid mode failed to initialize both LLMs. Falling back to Groq.")
                return self._fallback_to_groq()
                
            return self.groq_llm # Primary LLM for the QA chain
        
        else:
            try:
                return ChatOllama(
                    model=self.config['ollama']['model'],
                    temperature=self.config['ollama']['temperature']
                )
            except:
                logger.warning("Ollama not available. Falling back to Groq...")
                return self._fallback_to_groq()
    
    def _fallback_to_groq(self):
        """Fallback to Groq if primary provider fails"""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            logger.info("Falling back to Groq LLM")
            return ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024
            )
        else:
            logger.error("No LLM provider available. Please check configuration.")
            return None
    
    def auto_load_documents(self):
        """Automatically load all university documents from all data folders"""
        logger.info(f"Scanning data directories...")
        
        all_files = []
        for data_dir in self.data_dirs:
            logger.info(f"Scanning {data_dir}...")
            with open("debug_loading.log", "a") as f:
                f.write(f"DEBUG: Scanning directory: {data_dir}\n")
            
            file_patterns = [
                str(data_dir / "*.pdf"),
                str(data_dir / "*.txt"),
                str(data_dir / "*.docx"),
                str(data_dir / "*.doc"),
                str(data_dir / "*.csv")
            ]
            
            for pattern in file_patterns:
                # print(f"DEBUG: Glob pattern: {pattern}", flush=True)
                found = glob.glob(pattern)
                with open("debug_loading.log", "a") as f:
                     f.write(f"DEBUG: Pattern {pattern} found {len(found)} files\n")
                all_files.extend(found)
                if found:
                    logger.info(f"Found {len(found)} files in {data_dir}")
        
        if not all_files:
            logger.warning(f"No university documents found in any data directory")
            return False
        
        logger.info(f"Found {len(all_files)} document file(s)")
        
        # InMemoryVectorStore does not support persistence, so we always process files
        if False: # self._check_existing_vector_store(all_files):
            logger.info("Loading existing vector store...")
            # self.vector_store = Chroma(...)
            self._create_qa_chain()
            return True
        else:
            logger.info("Processing university documents...")
            return self.process_files(all_files)
    
    def _check_existing_vector_store(self, current_files):
        """Check if vector store exists and matches current files"""
        if not self.vector_store_path.exists():
            return False
        
        if not any(self.vector_store_path.iterdir()):
            return False
        
        try:
            temp_store = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embedding_model
            )
            
            existing_data = temp_store._collection.get()
            if not existing_data['metadatas']:
                return False
            
            existing_sources = set([m.get('source', '') for m in existing_data['metadatas']])
            current_sources = set([Path(f).name for f in current_files])
            
            return current_sources.issubset(existing_sources)
            
        except:
            return False
    
    def process_files(self, file_paths: List[str]):
        """Process and index university documents"""
        try:
            logger.info("Loading documents...")
            documents = []
            for i, file_path in enumerate(file_paths):
                docs = self._load_single_document(file_path)
                documents.extend(docs)
            
            if not documents:
                return False
            
            logger.info("Splitting into chunks...")
            chunks = self.split_documents(documents)
            
            logger.info("Creating in-memory vector database...")
            
            # Add documents in batches to avoid OOM
            batch_size = 5
            total_chunks = len(chunks)
            import gc
            import time
            logger.info(f"Processing {total_chunks} chunks in batches of {batch_size} with GC and sleep...")
            
            # First batch initializes the store
            if chunks:
                try:
                    logger.info(f"Initializing store with first batch (0-{batch_size})...")
                    self.vector_store = InMemoryVectorStore.from_documents(
                        documents=chunks[:batch_size],
                        embedding=self.embedding_model
                    )
                    gc.collect()
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Failed to initialize store: {e}")
                    return False

                # Remaining batches
                for i in range(batch_size, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    try:
                        self.vector_store.add_documents(batch)
                        if i % 100 == 0:
                            logger.info(f"Processed {i}/{total_chunks} chunks")
                            gc.collect()
                        time.sleep(0.05) # Yield CPU
                    except Exception as e:
                        logger.error(f"Failed to add batch {i}: {e}")
                        # Don't abort, try next batch? Or abort?
                        # If OOM, we probably can't continue well.
            else:
                 logger.warning("No chunks to process.")
            
            self._create_qa_chain()
            logger.info("Processing complete!")
            
            self.processed_files.update(file_paths)
            return True
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            import traceback
            trace = traceback.format_exc()
            logger.error(trace)
            with open("debug_loading.log", "a") as f:
                f.write(f"ERROR in process_files: {e}\n{trace}\n")
            return False
    
    def _load_single_document(self, file_path: str):
        """Load a single document"""
        try:
            file_name = Path(file_path).name
            with open("debug_loading.log", "a") as f:
                f.write(f"DEBUG: Loading {file_path}...\n")
            
            if file_path.endswith('.pdf'):
                # print(f"DEBUG: Loading PDF {file_path} with PyPDFLoader...", flush=True)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    # print(f"DEBUG: Loaded {len(docs)} pages.", flush=True)
                except Exception as e:
                    with open("debug_loading.log", "a") as f:
                        f.write(f"ERROR loading PDF {file_path}: {e}\n")
                    # print(f"DEBUG: PyPDFLoader failed: {e}", flush=True)
                    docs = []
            
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            
            elif file_path.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()

            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path, encoding='utf-8')
                docs = loader.load()
            
            else:
                return []
            
            with open("debug_loading.log", "a") as f:
                f.write(f"DEBUG: Loaded {len(docs)} docs from {file_path}\n")
            
            for doc in docs:
                doc.metadata.update({
                    "source": file_name,
                    "file_path": file_path,
                    "file_type": Path(file_path).suffix[1:].upper(),
                    "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return docs
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            with open("debug_loading.log", "a") as f:
                f.write(f"ERROR: Failed to load {file_path}: {e}\n")
            return []
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap'],
            separators=["\n\n", "\n", "。", "．", " ", ""],
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def _create_qa_chain(self):
        """Create the QA chain with UniMate-specific prompt"""
        if not self.llm:
            logger.error("LLM not initialized. Cannot create QA chain.")
            return
        
        # UniMate-specific prompt template
        prompt_template = """
        You are UniMate, a helpful University Assistant.

        ACRONYM GLOSSARY:
        - MIT: Management and Information Technology (usually at University of Kelaniya)
        - SJP: University of Sri Jayewardenepura
        - Kel / Kelaniya: University of Kelaniya
        - Moratuwa / Mor: University of Moratuwa
        - Peradeniya / Pera: University of Peradeniya
        - QS: Quantity Surveying
        - CS: Computer Science

        Rules:
        1. NO REPETITION: Do NOT repeat the student's question. Start your answer immediately.
        2. FORMATTING: 
           - Use BULLET POINTS for lists of data (e.g., Z-scores). 
           - Use PARAGRAPHS for normal conversation (e.g., greetings, explanations).
        3. SOURCE: 
           - FIRST, check the "University Data" below.
           - IF the answer is NOT in the data, use your own knowledge to answer, BUT ONLY if it relates to Sri Lankan Universities.
           - Rule of Thumb: If it's about Sri Lanka/Universities -> Answer. If it's about a foreign institution (like MIT USA), explain that you focus on Sri Lankan Universities (where MIT = Management and Information Technology).
        4. CONTENT RELEVANCE:
           - Check the "University Data" context provided below.
           - If the uploaded document or context is NOT related to university courses, Z-scores, or academic regulations, DO NOT ANSWER.
           - Instead, say: "I can only assist with university-related documents. This file does not appear to be relevant to Sri Lankan universities."
        5. NATURAL TONE & CITATIONS:
           - General Queries: Speak naturally as an expert. Start answers directly. Avoid "According to the provided data".
           - Document Analysis: IF the user asks about a specific uploaded file, you MAY say "Based on the document..." or "The document states..." to be precise.
        6. CHAT CONTINUITY:
           - IF the user provides a short input (e.g. just a district name like "Mullaitivu", "Gampaha"), DO NOT just greet them.
           - CHECK CHAT HISTORY to see if there was a previous unanswered question (e.g. "What is the Z-score for Bio?").
           - COMBINE the previous question with the new input (e.g. "Z-score for Bio in Mullaitivu") and ANSWER IT accurately.
           - THIS IS CRITICAL. Do not lose the context.

        University Data:
        {context}

        Chat History:
        {chat_history}

        Student: {question}
        UniMate:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config['retrieval']['top_k'],
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        
        # self.qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=retriever,
        #     memory=self.memory,
        #     combine_docs_chain_kwargs={"prompt": prompt},
        #     return_source_documents=True,
        #     # verbose=False # Causing attribute error?
        # )
        
        
        # Bypass standard chains due to Python 3.14 Pydantic incompatibility
        self.qa_chain = ManualRAG(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt
        )
        
        with open("debug_loading.log", "a") as f:
            f.write("DEBUG: QA Chain created successfully! (ManualRAG)\n")
    
    def clear_memory(self):
        """Clear the conversational memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Chatbot memory cleared.")
            return True
        return False

    def query(self, question: str):
        """Query the chatbot and track metrics"""
        if not self.llm:
            return {"error": "LLM not initialized. Check API keys or dependencies."}

        if not self.qa_chain:
            return {"error": "University documents not loaded. Please check data/ folder."}
        
        start_time = time.time()
        
        try:
            # Load memory
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            query_time = time.time() - start_time
            
            self.query_count += 1
            answer = result.get('answer', '')
            
            # Save to memory
            self.memory.save_context(
                {"input": question}, 
                {"answer": answer}
            )
            
            self.total_tokens += len(question) + len(answer)
            
            suggestions = self._generate_followup_questions(answer)
            
            sources = []
            for doc in result['source_documents']:
                sources.append({
                    "document": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "excerpt": doc.page_content[:200] + "...",
                    "similarity": "High"
                })
            
            return {
                "answer": answer,
                "suggestions": suggestions,
                "sources": sources,
                "llm_provider": self.config['llm_provider'],
                "model": self.config.get(self.config['llm_provider'], {}).get('model', 'Unknown'),
                "query_time": f"{query_time:.2f}s",
                "query_count": self.query_count
            }
            
        except Exception as e:
            query_time = time.time() - start_time
            return {
                "error": f"Query failed: {str(e)}",
                "query_time": f"{query_time:.2f}s"
            }
    
    def stream_query(self, question: str):
        """Stream the answer token-by-token"""
        print(f"DEBUG: stream_query called with: {question}", flush=True)
        
        # Hardcoded greeting to ensure exact compliance
        # But allow context continuation if it's a district name or short follow-up
        lower_q = question.strip().lower()
        districts = ["colombo", "gampaha", "kalutara", "matale", "kandy", "nuwara eliya", "galle", "matara", "hambantota", "jaffna", "kilinochchi", "mannar", "mullaitivu", "vavuniya", "trincomalee", "batticaloa", "ampara", "puttalam", "kurunegala", "anuradhapura", "polonnaruwa", "badulla", "monaragala", "kegalle", "ratnapura"]
        
        if lower_q in ["hi", "hello", "hey", "greetings"] and lower_q not in districts:
            greeting = "Hello! I am UniMate, your helpful assistant for information regarding Sri Lankan Universities. How can I assist you today?"
            yield {"type": "token", "token": greeting}
            # Also yield the final answer format for the fallback logic to catch if needed
            yield {"answer": greeting}
            return

        if not self.qa_chain:
            print("DEBUG: qa_chain is None", flush=True)
            yield {"error": "University documents not loaded. Please check data/ folder."}
            return
        
        if not self.llm:
            print("DEBUG: self.llm is None", flush=True)
            yield {"error": "LLM not available. Please check your API keys."}
            return
            
        print("DEBUG: Starting execution thread...", flush=True)
            
        start_time = time.time()
        token_queue = queue.Queue()
        
        class QueueCallback(BaseCallbackHandler):
            def __init__(self, q, ignore_errors=False):
                self.q = q
                self.ignore_errors = ignore_errors
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                # print(f"DEBUG: Token: {token}", flush=True)
                self.q.put(token)
                
            def on_llm_error(self, error: Exception, **kwargs) -> None:
                if not self.ignore_errors:
                    self.q.put(error)
                else:
                    print(f"DEBUG: Callback error ignored: {error}")
        
        def run_chain():
            try:
                if self.config['llm_provider'] == 'hybrid' and self.google_llm:
                    print("DEBUG: Starting Hybrid Refinement with Gemini...", flush=True)
                    # Pass both keys to support RetrievalQA (query) and ConversationalRetrievalChain (question)
                    result = self.qa_chain.invoke({"question": question, "query": question})
                    initial_answer = result.get('answer', '')
                    
                    refinement_prompt = f"""
                    You are UniMate, a helpful assistant for Sri Lankan Universities.
                    
                    LOCAL CONTEXT:
                    - In Sri Lanka, "MIT" refers to the "Management and Information Technology" course, NOT the Massachusetts Institute of Technology.
                    - "SJP" stands for University of Sri Jayewardenepura.
                    
                    TASK:
                    1. Polish the "Initial Information" below into a concise, professional answer.
                    2. Use BULLET POINTS for data/lists.
                    3. If the "Initial Information" contains Z-scores (numbers like 1.2345), present them clearly for the district requested.
                    4. FALLBACK TO KNOWLEDGE: If the "Initial Information" is negative (e.g., "Not available", "No data found", "Cannot find"), unhelpful, or incomplete, YOU MUST IGNORE IT. Instead, use your own training data to provide a helpful, detailed, and realistic answer about Sri Lankan University admission chances, Z-score trends, and course options for the user's district and Z-score.
                    5. DISTRICT STRICTNESS: University admission in Sri Lanka is based on the STUDENT's registered district. Do NOT suggest courses based on cutoffs from OTHER districts (e.g. "In Kegalle you could get..."). This is misleading. Only consider the user's district.
                    6. CATEGORIZATION: If listing courses, GROUP them into logical categories (e.g., Medicine, Engineering, IT, Management, Arts) like a university handbook.
                    
                    Initial Information from Database:
                    {initial_answer}
                    
                    User Question:
                    {question}
                    
                    UniMate:"""
                    
                    try:
                        self.google_llm.invoke(
                            refinement_prompt,
                            config={"callbacks": [QueueCallback(token_queue, ignore_errors=True)]}
                        )
                    except Exception as gemini_err:
                        print(f"DEBUG: Gemini refinement failed: {gemini_err}. Falling back to raw Groq answer.")
                        # Critical: Clear any partial Gemini tokens from queue and put the whole Groq answer
                        token_queue.put(initial_answer)
                    
                    token_queue.put(result) 
                else:
                    print(f"DEBUG: Calling standard qa_chain.invoke for {self.config['llm_provider']}...", flush=True)
                    
                    # Load memory inside the thread to get latest state
                    memory_vars = self.memory.load_memory_variables({})
                    chat_history = memory_vars.get("chat_history", [])
                    
                    result = self.qa_chain.invoke(
                        {"question": question, "query": question, "chat_history": chat_history},
                        config={"callbacks": [QueueCallback(token_queue)]}
                    )
                    token_queue.put(result)
            except Exception as e:
                print(f"DEBUG: Exception in run_chain: {e}")
                token_queue.put(e)

        thread = threading.Thread(target=run_chain)
        thread.start()
        
        answer_accumulated = ""
        
        while True:
            try:
                # print("DEBUG: Waiting for token...")
                item = token_queue.get(timeout=120)
                
                if isinstance(item, str):
                    answer_accumulated += item
                    yield {"type": "token", "token": item}
                elif isinstance(item, dict):
                    # Support multiple output keys from different chain types
                    ans = item.get('answer') or item.get('result') or item.get('output')
                    
                    if ans:
                        # Fallback: if zero tokens were streamed, send the whole answer now
                        if not answer_accumulated:
                            answer_accumulated = ans
                            yield {"type": "token", "token": ans}
                        
                        # Save to memory after full answer is generated
                        self.memory.save_context(
                            {"input": question}, 
                            {"answer": ans}
                        )
                    query_time = time.time() - start_time
                    self.query_count += 1
                    self.total_tokens += len(question) + len(answer_accumulated)
                    
                    suggestions = self._generate_followup_questions(answer_accumulated)
                    
                    sources = []
                    if 'source_documents' in item:
                        for doc in item['source_documents']:
                            sources.append({
                                "document": doc.metadata.get('source', 'Unknown'),
                                "page": doc.metadata.get('page', 'N/A'),
                                "excerpt": doc.page_content[:200] + "...",
                                "similarity": "High"
                            })
                            
                    final_response = {
                        "type": "final",
                        "answer": answer_accumulated,
                        "suggestions": suggestions,
                        "sources": sources,
                        "llm_provider": self.config['llm_provider'],
                        "model": self.config.get(self.config['llm_provider'], {}).get('model', 'Unknown'),
                        "query_time": f"{query_time:.2f}s",
                        "query_count": self.query_count
                    }
                    yield final_response
                    break
                elif isinstance(item, Exception):
                    yield {"error": str(item)}
                    break
                    
            except queue.Empty:
                yield {"error": "Timeout waiting for response"}
                break

    def _generate_followup_questions(self, answer: str) -> List[str]:
        """Generate 3 relevant follow-up questions based on the answer"""
        try:
            prompt = f"""Based on the answer below, suggest 3 short, relevant follow-up questions a user might ask.
            
            Answer: {answer}
            
            Format:
            - Provide ONLY the 3 questions
            - Separate by newlines
            - No numbering or bullets
            - Keep them concise (under 15 words)
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            
            final_questions = []
            for q in questions:
                clean_q = q.lstrip('1234567890.- ').strip()
                if clean_q and '?' in clean_q:
                    final_questions.append(clean_q)
            
            return final_questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

    def get_stats(self):
        """Return chatbot statistics"""
        return {
            "query_count": self.query_count,
            "total_tokens": self.total_tokens,
            "llm_provider": self.config.get('llm_provider', 'Unknown'),
            "model": self.config.get(self.config.get('llm_provider', 'google'), {}).get('model', 'Unknown')
        }
