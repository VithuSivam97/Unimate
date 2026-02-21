import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, Paperclip, StopCircle, RefreshCw, AlertCircle } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown'; // We might need to install this: npm install react-markdown

// Helper for Fetch Event Handling
import { fetchEventSource } from '@microsoft/fetch-event-source';

const ChatInterface = ({ currentSessionId }) => {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    // Reset session on mount (refresh)
    useEffect(() => {
        const resetSession = async () => {
            try {
                await axios.get('http://127.0.0.1:8000/reset');
                console.log('Session reset successfully');
            } catch (error) {
                console.error('Failed to reset session:', error);
            }
        };
        resetSession();
    }, []);

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('files', file);

        // Optimistic UI update
        const uploadMsgId = Date.now();
        setMessages(prev => [...prev, { role: 'assistant', content: 'Uploading document...', id: uploadMsgId }]);
        setIsLoading(true);

        try {
            await axios.post('http://127.0.0.1:8000/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setMessages(prev => prev.map(m =>
                m.id === uploadMsgId
                    ? { ...m, content: `Document "${file.name}" uploaded successfully! I have processed it. You can now ask questions about it.` }
                    : m
            ));
        } catch (err) {
            console.error(err);
            setMessages(prev => prev.map(m =>
                m.id === uploadMsgId
                    ? { ...m, content: `Failed to upload document: ${err.message}` }
                    : m
            ));
        } finally {
            setIsLoading(false);
            // Reset file input
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setError(null);

        // Create a placeholder for assistant message
        const assistantMessageId = Date.now();
        setMessages(prev => [...prev, { role: 'assistant', content: '', id: assistantMessageId }]);

        try {
            // Connect to Streaming API
            let accumulatedResponse = "";

            await fetchEventSource('http://127.0.0.1:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage.content }),
                onmessage(msg) {
                    try {
                        const data = JSON.parse(msg.data);

                        if (data.type === 'token') {
                            accumulatedResponse += data.token;

                            setMessages(prev => prev.map(m =>
                                m.id === assistantMessageId
                                    ? { ...m, content: accumulatedResponse }
                                    : m
                            ));
                        } else if (data.type === 'final') {
                            // Finalize with suggestions/sources
                            setMessages(prev => prev.map(m =>
                                m.id === assistantMessageId
                                    ? {
                                        ...m,
                                        content: data.answer,
                                        sources: data.sources,
                                        suggestions: data.suggestions
                                    }
                                    : m
                            ));
                        } else if (data.error) {
                            setError(data.error);
                        }
                    } catch (err) {
                        console.error("Parse error", err);
                    }
                },
                onclose() {
                    setIsLoading(false);
                },
                onerror(err) {
                    setError("Connection failed");
                    setIsLoading(false);
                    throw err; // rethrow to stop
                }
            });

        } catch (err) {
            console.error(err);
            setError("Failed to generate response");
            setIsLoading(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-slate-50 overflow-hidden">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-8">
                <div className="max-w-3xl mx-auto space-y-6">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center space-y-6 animate-in fade-in duration-500 mt-20">
                            <h2 className="text-3xl font-bold text-gray-800 tracking-tight">How can I help with University information today?</h2>
                            <p className="max-w-lg text-lg text-gray-600">
                                Ask about Z-scores, courses, or specific university details. I'll provide citations for every answer.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {/* Message Bubble */}
                            <div className={`max-w-[85%] space-y-2`}>
                                <div className={`
                    p-4 rounded-2xl shadow-sm text-sm leading-relaxed
                    ${msg.role === 'user'
                                        ? 'bg-gray-800 text-white rounded-tr-none'
                                        : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'}
                `}>
                                    {msg.role === 'assistant' ? (
                                        // Simple rendering for now, can replace with ReactMarkdown
                                        <div className="whitespace-pre-wrap">{msg.content || <span className="animate-pulse">Thinking...</span>}</div>
                                    ) : (
                                        msg.content
                                    )}
                                </div>

                                {/* Sources (Removed) */}
                                {/* Sources removed as per user request */}

                                {/* Suggestions */}
                                {msg.suggestions && msg.suggestions.length > 0 && (
                                    <div className="flex flex-wrap gap-2 pt-1">
                                        {msg.suggestions.map((sugg, sIdx) => (
                                            <button
                                                key={sIdx}
                                                onClick={() => {
                                                    setInput(sugg);
                                                    // Optional: Auto submit
                                                    // handleSubmit({ preventDefault: () => {} });
                                                }}
                                                className="text-xs bg-white border border-slate-200 px-3 py-1.5 rounded-full text-slate-600 hover:text-primary hover:border-primary transition-colors hover:shadow-sm"
                                            >
                                                {sugg}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    {error && (
                        <div className="flex justify-center">
                            <div className="bg-red-50 text-red-600 text-sm px-4 py-2 rounded-full flex items-center gap-2">
                                <AlertCircle size={16} />
                                {error}
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white border-t border-slate-100">
                <div className="max-w-4xl mx-auto">
                    <form onSubmit={handleSubmit} className="relative flex items-center bg-slate-50 border border-slate-200 rounded-full px-2 py-2 focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all shadow-sm focus-within:shadow-md">

                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="p-2 text-gray-500 hover:text-gray-700 transition-colors rounded-full hover:bg-gray-100"
                            title="Upload Document"
                            disabled={isLoading}
                        >
                            <Paperclip size={20} />
                        </button>
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            accept=".pdf"
                            onChange={handleFileUpload}
                        />

                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask UniMate a question..."
                            className="flex-1 bg-transparent border-none focus:ring-0 px-4 py-2 text-slate-800 placeholder-slate-400 outline-none"
                            disabled={isLoading}
                        />

                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className={`
                p-2 rounded-full transition-all duration-200
                ${input.trim() && !isLoading
                                    ? 'bg-gray-900 text-white hover:bg-black shadow-md transform hover:scale-105'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'}
              `}
                        >
                            {isLoading ? <StopCircle size={20} className="animate-pulse" /> : <Send size={20} />}
                        </button>
                    </form>
                    <div className="text-center mt-2">
                        <span className="text-[10px] text-slate-400">AI can make mistakes. Verify important information.</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
