
import os
import sys
import yaml
import glob
import time
import threading
import queue
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    print("Warning: Could not import ChatOpenAI. OpenAI support will be unavailable.")
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader
)
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from history_manager import HistoryManager
# from user_manager import UserManager
# import extra_streamlit_components as stx
import datetime


# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
# Load environment variables
load_dotenv(override=True)

class GazetteChatbot:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Gazette Chatbot with Groq support"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize paths
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.vector_store_path = Path(self.config['paths']['vector_store'])
        
        # Track costs and performance
        self.query_count = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        # Initialize models
        self.embedding_model = self._init_embedding_model()
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
            self.auto_load_gazettes()
    
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
            st.error(f"Failed to load embedding model: {e}")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _init_llm(self):
        """Initialize LLM based on provider choice"""
        provider = self.config['llm_provider'].lower()
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.warning("⚠️ Groq API key not found in .env file")
                st.info("Getting FREE API key: https://console.groq.com/keys")
                return None
            
            model_name = self.config['groq']['model']
            groq_models = {
                "llama-3.3-70b-versatile": "Llama 3.3 70B (Current Best)",  # ADD THIS
            "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Efficient)",  # ADD THIS
            "llama3-70b-8192": "Llama 3.1 70B (Deprecated)",  # Keep but mark deprecated
                "mixtral-8x7b-32768": "Mixtral 8x7B (Good balance)",
                "gemma2-9b-it": "Gemma 2 9B (Fastest)",
                "llama3-8b-8192": "Llama 3.1 8B (Efficient)"
            }
            
            if model_name not in groq_models:
                st.warning(f"Model {model_name} not recognized. Using Llama 3.1 70B")
                model_name = "llama-3.3-70b-versatile"
            
            st.info(f"🚀 Using Groq: {groq_models[model_name]}")
            
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=self.config['groq']['temperature'],
                max_tokens=self.config['groq']['max_tokens'],
                streaming=True
            )
        
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.warning("OpenAI API key not found. Falling back to Groq...")
                return self._fallback_to_groq()
            
            return ChatOpenAI(
                model=self.config['openai']['model'],
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens'],
                openai_api_key=api_key,
                streaming=True
            )
        
        else:
            try:
                return ChatOllama(
                    model=self.config['ollama']['model'],
                    temperature=self.config['ollama']['temperature']
                )
            except:
                st.warning("Ollama not available. Falling back to Groq...")
                return self._fallback_to_groq()
    
    def _fallback_to_groq(self):
        """Fallback to Groq if primary provider fails"""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.info("🔄 Falling back to Groq LLM")
            return ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024
            )
        else:
            st.error("❌ No LLM provider available. Please check configuration.")
            return None
    
    def auto_load_gazettes(self):
        """Automatically load all gazette files from data folder"""
        st.info(f"🔍 Scanning {self.data_dir} for gazette files...")
        
        file_patterns = [
            str(self.data_dir / "*.pdf"),
            str(self.data_dir / "*.txt"),
            str(self.data_dir / "*.docx"),
            str(self.data_dir / "*.doc")
        ]
        
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            st.warning(f"📭 No gazette files found in {self.data_dir}")
            st.info(f"💡 Please add PDF/TXT/DOCX files to: {self.data_dir.absolute()}")
            return False
        
        st.success(f"📁 Found {len(all_files)} gazette file(s)")
        
        if self._check_existing_vector_store(all_files):
            st.info("✅ Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embedding_model
            )
            self._create_qa_chain()
            return True
        else:
            st.info("🔄 Processing gazette files...")
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
        """Process and index gazette files"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Loading documents...")
            documents = []
            for i, file_path in enumerate(file_paths):
                progress_bar.progress((i) / len(file_paths) * 0.3)
                docs = self._load_single_document(file_path)
                documents.extend(docs)
            
            if not documents:
                return False
            
            status_text.text("✂️  Splitting into chunks...")
            chunks = self.split_documents(documents)
            progress_bar.progress(0.6)
            
            status_text.text("💾 Creating vector database...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=str(self.vector_store_path)
            )
            progress_bar.progress(0.9)
            
            self._create_qa_chain()
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            self.processed_files.update(file_paths)
            
            st.success(f"✅ Successfully processed {len(documents)} document chunks")
            
            with st.expander("📊 Processing Statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", len(file_paths))
                with col2:
                    st.metric("Document Chunks", len(documents))
                with col3:
                    st.metric("Vector Chunks", len(chunks))
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing files: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _load_single_document(self, file_path: str):
        """Load a single document"""
        try:
            file_name = Path(file_path).name
            
            if file_path.endswith('.pdf'):
                try:
                    loader = UnstructuredPDFLoader(file_path, mode="elements")
                    docs = loader.load()
                except:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
            
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            
            elif file_path.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            
            else:
                return []
            
            for doc in docs:
                doc.metadata.update({
                    "source": file_name,
                    "file_path": file_path,
                    "file_type": Path(file_path).suffix[1:].upper(),
                    "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return docs
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file_path}: {e}")
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
        """Create the QA chain with gazette-specific prompt"""
        if not self.llm:
            st.error("LLM not initialized. Cannot create QA chain.")
            return
        
        # Gazette-specific prompt template
        prompt_template = """
        You are a Gazette Document Expert Assistant. Your task is to answer questions based ONLY on the provided gazette excerpts.
        
        CRITICAL RULES:
        1. Use ONLY information from the provided context below
        2. If the context doesn't contain relevant information, say "Based on the provided gazette documents, I cannot find information about [topic]"
        3. Be precise and factual - gazettes are official documents
        4. When citing information, mention the source document name
        5. Format dates, numbers, and legal terms clearly

        Context from gazettes:
        {context}

        Question: {question}

        Provide a clear, concise answer based on the gazette context:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.config['retrieval']['top_k']
            }
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str):
        """Query the chatbot and track metrics"""
        if not self.qa_chain:
            return {"error": "Gazette documents not loaded. Please check data/ folder."}
        
        if not self.llm:
            return {"error": "LLM not available. Please check your API keys."}
        
        start_time = time.time()
        
        try:
            # For ConversationalRetrievalChain, we pass the question as "question"
            # Memory handles chat_history automatically
            result = self.qa_chain.invoke({"question": question})
            query_time = time.time() - start_time
            
            self.query_count += 1
            # Result contains 'answer' key now instead of 'result'
            answer = result.get('answer', '')
            self.total_tokens += len(question) + len(answer)
            
            # Generate follow-up suggestions
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
        if not self.qa_chain:
            yield {"error": "Gazette documents not loaded. Please check data/ folder."}
            return
        
        if not self.llm:
            yield {"error": "LLM not available. Please check your API keys."}
            return
            
        start_time = time.time()
        
        # Queue for tokens
        token_queue = queue.Queue()
        
        # Callback to put tokens in queue
        class QueueCallback(BaseCallbackHandler):
            def __init__(self, q):
                self.q = q
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.q.put(token) # Put string token
                
            def on_llm_end(self, *args, **kwargs) -> None:
                pass
                
            def on_llm_error(self, error: Exception, **kwargs) -> None:
                self.q.put(error)
        
        # Function to run chain in separate thread
        def run_chain():
            try:
                # We attach the callback to the invoke call
                result = self.qa_chain.invoke(
                    {"question": question},
                    config={"callbacks": [QueueCallback(token_queue)]}
                )
                # Put the final result dict in the queue
                token_queue.put(result)
            except Exception as e:
                token_queue.put(e)

        # Start thread
        thread = threading.Thread(target=run_chain)
        thread.start()
        
        # Yield tokens from queue
        answer_accumulated = ""
        
        while True:
            try:
                # Wait for next item
                item = token_queue.get(timeout=120) # 2 min timeout safety
                
                if isinstance(item, str):
                    # It's a token
                    answer_accumulated += item
                    yield item
                elif isinstance(item, dict) and "answer" in item:
                    # It's the final result
                    query_time = time.time() - start_time
                    self.query_count += 1
                    
                    # Update metrics
                    self.total_tokens += len(question) + len(answer_accumulated)
                    
                    suggestions = self._generate_followup_questions(item['answer'])
                    
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
                        "answer": item['answer'],
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
            
            # Use the existing LLM to generate suggestions
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse lines
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Filter and cleanup
            final_questions = []
            for q in questions:
                # Remove common prefixes like "1. " or "- "
                clean_q = q.lstrip('1234567890.- ').strip()
                if clean_q and '?' in clean_q:
                    final_questions.append(clean_q)
            
            # Return top 3
            return final_questions[:3]
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []
    
    def get_stats(self):
        """Get statistics about the system"""
        stats = {
            "llm_provider": self.config['llm_provider'],
            "model": self.config.get(self.config['llm_provider'], {}).get('model', 'Unknown'),
            "embedding_model": self.config['embeddings']['model'],
            "data_folder": str(self.data_dir.absolute()),
            "query_count": self.query_count,
            "total_tokens": self.total_tokens,
            "uptime": f"{(time.time() - self.start_time):.0f}s"
        }
        
        if self.vector_store:
            try:
                count = self.vector_store._collection.count()
                stats["documents_loaded"] = count
                stats["status"] = "Ready"
            except:
                stats["status"] = "Vector store error"
        else:
            stats["status"] = "Not loaded"
        
        return stats

def run_chat_interface():
    """Main Streamlit application logic"""
    # Page config moved to wrapper main()
    
    # Title
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.image("assets/logo.png", width=70)
    with col2:
        st.title("Gazette Chatbot")
    st.markdown("⚡ Powered by Groq - Super fast responses")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.initialized = False
        st.session_state.messages = []
    
    # Initialize history manager
    if 'history_manager' not in st.session_state:
        st.session_state.history_manager = HistoryManager()
        # Load most recent session or create new
        sessions = st.session_state.history_manager.get_all_sessions()
        if sessions:
            st.session_state.current_session_id = sessions[0]['id']
            st.session_state.messages = st.session_state.history_manager.get_session(sessions[0]['id']).get('messages', [])
        else:
            new_id = st.session_state.history_manager.create_new_session()
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
    
    # Simple sidebar
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.header("⚙️ Configuration")
        
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            provider = config.get('llm_provider', 'groq').upper()
            model = config.get(provider.lower(), {}).get('model', 'Unknown')
            
            st.markdown(f"**Provider:** {provider}")
            st.markdown(f"**Model:** `{model}`")
            
        except Exception as e:
            st.error(f"Config error: {e}")
        
        st.markdown("---")
        
        # Data folder status
        data_dir = Path("./data")
        if data_dir.exists():
            files = list(data_dir.glob("*"))
            if files:
                st.success(f"✅ Found {len(files)} file(s)")
            else:
                st.warning("📭 Folder is empty")
        else:
            st.error("❌ data/ folder not found!")
            
        st.markdown("---")
        st.header("🕒 Chat History")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            new_id = st.session_state.history_manager.create_new_session()
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.rerun()
            
        # Session List
        sessions = st.session_state.history_manager.get_all_sessions()
        for session in sessions:
            title = session['title']
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                # Highlight current session
                if session['id'] == st.session_state.current_session_id:
                   st.button(f"📂 {title}", key=session['id'], disabled=True, use_container_width=True)
                else:
                    if st.button(f"📄 {title}", key=session['id'], use_container_width=True):
                        st.session_state.current_session_id = session['id']
                        st.session_state.messages = st.session_state.history_manager.get_session(session['id']).get('messages', [])
                        st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{session['id']}", help="Delete this chat"):
                    st.session_state.history_manager.delete_session(session['id'])
                    
                    # If deleted session was current, switch to another
                    if session['id'] == st.session_state.current_session_id:
                        updated_sessions = st.session_state.history_manager.get_all_sessions()
                        if updated_sessions:
                            next_session = updated_sessions[0]
                            st.session_state.current_session_id = next_session['id']
                            st.session_state.messages = st.session_state.history_manager.get_session(next_session['id']).get('messages', [])
                        else:
                            # No sessions left, create new
                            new_id = st.session_state.history_manager.create_new_session()
                            st.session_state.current_session_id = new_id
                            st.session_state.messages = []
                    
                    st.rerun()
    
    # Main content - SIMPLIFIED
    if not st.session_state.initialized:
        st.info("Initializing Gazette Chatbot...")
        
        with st.spinner("Loading..."):
            try:
                chatbot = GazetteChatbot()
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
                st.success("✅ System ready!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
    else:
        # Simple chat interface
        st.markdown("## 💬 Chat")
        
        # Helper to process query
        def handle_query(prompt_text):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            with st.chat_message("user"):
                st.markdown(prompt_text)
            
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use the streaming method
                try:
                    # Initial state
                    message_placeholder.markdown("Thinking...")
                    
                    generator = st.session_state.chatbot.stream_query(prompt_text)
                    final_data = None
                    
                    for item in generator:
                        if isinstance(item, str):
                            full_response += item
                            message_placeholder.markdown(full_response + "▌")
                        elif isinstance(item, dict):
                            if "error" in item:
                                message_placeholder.error(item["error"])
                                return
                            elif "type" in item and item["type"] == "final":
                                final_data = item
                                message_placeholder.markdown(item["answer"])
                    
                    if final_data:
                        # Append to state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_data["answer"],
                            "sources": final_data.get("sources", []),
                            "suggestions": final_data.get("suggestions", [])
                        })
                        
                        # Save to history
                        st.session_state.history_manager.save_session(
                            st.session_state.current_session_id,
                            st.session_state.messages
                        )
                        
                        # Rerun to update the full UI with sources and buttons
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

        # Display messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    st.markdown("### 📚 Sources")
                    for source in message["sources"]:
                        with st.expander(f"📚 Source: {source['document']} (Page {source['page']})"):
                            st.markdown(f"**Excerpt:**\n{source['excerpt']}")
            
            # If it's the last message and it's from assistant, show suggestions
            if i == len(st.session_state.messages) - 1 and message["role"] == "assistant":
                if "suggestions" in message and message["suggestions"]:
                    st.markdown("---")
                    st.markdown("### 💡 Suggested Follow-up Questions:")
                    cols = st.columns(len(message["suggestions"]))
                    for idx, suggestion in enumerate(message["suggestions"]):
                        if cols[idx].button(suggestion, key=f"sugg_{i}_{idx}"):
                            handle_query(suggestion)
        
        # Chat input
        if prompt := st.chat_input("Ask about gazette documents..."):
            handle_query(prompt)

def main():
    st.set_page_config(
        page_title="Gazette Chatbot",
        page_icon="assets/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Direct access check - bypassing all auth
    run_chat_interface()

if __name__ == "__main__":
    main()