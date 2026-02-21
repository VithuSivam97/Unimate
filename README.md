# UniMate - University Assistant

UniMate is a powerful RAG (Retrieval-Augmented Generation) chatbot system designed to answer questions based on university documents, such as Z-scores, handbooks, and official gazettes. It features a modern React-based frontend and a robust FastAPI backend, leveraging LangChain and Groq for fast, accurate AI responses.

## 🚀 Key Features

- **Retrieval-Augmented Generation (RAG)**: Answers questions with precision by sourcing information directly from uploaded documents.
- **Multi-LLM Support**: Configurable to use **Groq** (Llama 3), **OpenAI**, or **Ollama**.
- **Modern Web Interface**: Built with React 19, Vite, and Tailwind CSS for a smooth, responsive user experience.
- **FastAPI Backend**: High-performance asynchronous API handling chat, file uploads, and document processing.
- **Streaming Responses**: Real-time token streaming for instantaneous feedback.
- **Document Management**: Support for PDF, TXT, and Excel/CSV document processing.
- **Smart Suggestions**: Automatically generates follow-up questions based on the chat context.
- **Citations**: Provides clear references to the sources used for each answer.

## 🛠️ Tech Stack

- **Frontend**: React 19, Vite, Tailwind CSS, Lucide React, Framer Motion.
- **Backend**: Python 3.9+, FastAPI, LangChain, Uvicorn.
- **AI/LLM**: Groq (Llama 3), OpenAI (optional).
- **Data Storage**: ChromaDB (Vector Store), CSV/PDF/Excel.

## ⚙️ Prerequisites

- **Python** 3.9 or higher
- **Node.js** 18 or higher
- **Git**

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/VithuSivam97/Unimate.git
cd Unimate
```

### 2. Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements_fixed.txt
   ```
3. Configure environment variables:
   Create a `.env` file in the root directory:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   # Optional: OPENAI_API_KEY=your_openai_key_here
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

## 🏃 Running the Application

To run the full system, you will need to open two terminal windows.

### Terminal 1: Backend
From the root directory:
```bash
python -m uvicorn backend.api:app --reload
```
*The API will be available at `http://localhost:8000`*

### Terminal 2: Frontend
From the `frontend` directory:
```bash
npm run dev
```
*The web interface will be available at `http://localhost:5173`*

## 📂 Project Structure

- `backend/`: Core logic, API endpoints, and chatbot implementation.
- `frontend/`: React components, styles, and web application logic.
- `vector_store_clean/`: Persistent storage for document embeddings.
- `config.yaml`: Global configuration for LLMs and retrieval settings.
- `auth_utils.py`, `user_manager.py`: Utilities for authentication and user management.

## 🧪 Troubleshooting

- **API Connection**: Ensure the backend is running before starting the frontend.
- **Dependency Issues**: If you encounter errors, try reinstalling with `pip install -r requirements_fixed.txt` to ensure compatible versions.
- **Missing API Key**: Verify your `.env` file contains a valid `GROQ_API_KEY`.

## 📄 License

This project is licensed under the MIT License.
