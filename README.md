# UniMate - University Assistant

A powerful RAG (Retrieval-Augmented Generation) chatbot designed to answer questions based on University documents (Z-scores, Handbooks). Built with **Streamlit/FastAPI**, **LangChain**, and **Groq**, this application provides fast, accurate answers with citations.

## 🚀 Features

*   **Multi-LLM Support**: Seamlessly switch between **Groq** (Llama 3), **OpenAI**, and **Ollama**.
*   **Document QA**: Upload PDF, TXT, and DOCX files to query your own knowledge base.
*   **Smart Citations**: Answers include exact source references, page numbers, and excerpts.
*   **Streaming Responses**: Real-time token streaming for a responsive user experience.
*   **Chat History**: Auto-saves your chat sessions so you can revisit them later.
*   **Follow-up Suggestions**: intelligently suggests 3 relevant follow-up questions after every answer.
*   **Local Vector Store**: Uses ChromaDB for efficient and private document embedding storage.
*   **Memory**: Context-aware conversations that remember previous interactions.

## 🛠️ Prerequisites

*   Python 3.9 or higher
*   Git

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/archchika02/gazette-chatbot.git
    cd gazette-chatbot
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements_fixed.txt
    ```

## ⚙️ Configuration

### 1. Environment Variables
Create a `.env` file in the root directory and add your API keys:

```ini
# .env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...  # Optional, if using OpenAI
```

> **Note:** You can get a free Groq API key from [Groq Console](https://console.groq.com/keys).

### 2. Application Config
You can modify `config.yaml` to change models, chunking settings, or retrieval parameters:

```yaml
llm_provider: groq  # options: groq, openai, ollama

groq:
  model: llama-3.3-70b-versatile
  temperature: 0.1

retrieval:
  top_k: 7
```

## 🏃‍♂️ How to Run (New Architecture)
The application has been migrated to **React + FastAPI**. You need to run two terminals.

### 1. Start Backend
```bash
.\venv\Scripts\uvicorn backend.api:app --reload
```
> Server will start at `http://localhost:8000`

### 2. Start Frontend
```bash
cd frontend
npm run dev
```
> UI will start at `http://localhost:5173`

*(Legacy Streamlit app is renamed to `main_legacy.py`)*

## 📂 Project Structure

```
gazette-chatbot/
├── assets/                 # Images and static assets
├── data/                   # Place your source documents here
├── vector_store/           # ChromaDB persistence directory (auto-generated)
├── config.yaml             # Configuration settings
├── main.py                 # Main Streamlit application
├── history_manager.py      # Handles chat session persistence
├── requirements_fixed.txt  # Project dependencies
└── README.md               # This file
```

## 🧩 Troubleshooting

*   **"Meta Tensor Error"**: This is usually due to conflicting versions of `transformers` or `accelerate`. Ensure you installed strictly from `requirements_fixed.txt`.
*   **"No gazette files found"**: Make sure you have added supported files (.pdf, .txt, .docx) to the `data/` directory.


## 👥 Team Members

*   **T. Rahitha**
*   **T. Archchika** ([@archchika02](https://github.com/archchika02))
*   **L.J. Thilukshika**
*   **S. Thushanthi**

## 📄 License

[MIT License](LICENSE)