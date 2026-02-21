from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import shutil
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Import the refactored chatbot
# Ensure backend directory is in python path or use relative import if running as module
# Triggering reload for new data (Error Logic)
try:
    from backend.chatbot import UniMateChatbot
except ImportError:
    from chatbot import UniMateChatbot  # For running directly inside backend/

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    # try:
    with open("debug_init.log", "a") as f:
        f.write("LIFESPAN STARTING...\n")
    print("Initializing UniMate Chatbot...")
    chatbot = UniMateChatbot()
    with open("debug_init.log", "a") as f:
        f.write("UniMate Initialized!\n")
    print("UniMate Initialized!")
    yield
    # Shutdown logic if needed

app = FastAPI(title="UniMate API", lifespan=lifespan)

# CORS Configuration
origins = [
    "http://localhost:5173",  # React Frontend (Vite)
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5174",  # Added for fallback port
    "http://localhost:5175",  # Added for second fallback port
    "*"                       # Allow all for debugging
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    suggestions: List[str]
    sources: List[dict]
    query_time: str

@app.get("/")
async def root():
    global chatbot
    status = "running"
    if chatbot:
        status += " - Chatbot READY"
    else:
        status += " - Chatbot NONE"
        status += " - Chatbot NONE"
    return {"status": "ok", "message": "UniMate API is running", "chatbot_state": str(chatbot)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Use streaming response
    return StreamingResponse(
        stream_generator(request.message),
        media_type="text/event-stream"
    )

def stream_generator(message: str):
    """Generator for streaming response"""
    try:
        print(f"DEBUG: stream_generator called for message: {message}", flush=True)
        generator = chatbot.stream_query(message)
        print("DEBUG: Generator created", flush=True)
        for item in generator:
            # print(f"DEBUG: Generator yielded: {item}", flush=True)
            # Format as SSE (Server-Sent Events)
            json_data = json.dumps(item)
            yield f"data: {json_data}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/history")
async def get_history():
    # TODO: Implement history retrieval if needed by frontend
    # For now, frontend can manage its own history or we can expose HistoryManager
    return {"history": []}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    saved_files = []
    try:
        data_dir = chatbot.data_dirs[0]
        
        for file in files:
            file_path = data_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
        
        # Trigger processing
        success = chatbot.process_files(saved_files)
        
        if success:
            return {"status": "success", "message": f"Processed {len(saved_files)} files"}
        else:
            return {"status": "error", "message": "Failed to process files"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return chatbot.get_stats()

@app.get("/reset")
async def reset_chat():
    global chatbot
    if not chatbot:
        return {"status": "error", "message": "Chatbot not initialized"}
    
    chatbot.clear_memory()
    return {"status": "success", "message": "Chat history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
