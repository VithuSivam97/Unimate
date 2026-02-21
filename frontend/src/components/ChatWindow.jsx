import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatWindow({ messages, onSend }) {
    const [input, setInput] = useState("");
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = () => {
        if (!input.trim()) return;
        onSend(input);
        setInput("");
    };

    return (
        <div className="flex flex-col h-screen bg-gray-100">
            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto scroll-smooth">
                <div className="max-w-3xl mx-auto w-full p-6">
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}
                        >
                            <div
                                className={`mb-4 max-w-[85%] px-5 py-3 rounded-2xl shadow-sm overflow-hidden ${msg.sender === "bot"
                                    ? "bg-white text-gray-800 self-start rounded-tl-sm border border-gray-200"
                                    : "bg-gray-800 text-white self-end rounded-tr-sm"
                                    }`}
                            >
                                {msg.sender === "bot" ? (
                                    <div className="whitespace-pre-wrap font-sans text-sm leading-relaxed">{msg.text}</div>
                                ) : (
                                    msg.text
                                )}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Box */}
            <div className="border-t border-gray-300 bg-white p-4">
                <div className="max-w-3xl mx-auto w-full flex">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type a message..."
                        className="flex-1 border border-gray-300 rounded-md p-2 mr-2 focus:outline-none focus:ring-2 focus:ring-gray-400"
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                        autoFocus
                    />
                    <button
                        onClick={handleSend}
                        className="bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-900"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div >
    );
}
