import React from 'react';
import { Plus, MessageSquare, Trash2, Settings, X, BookOpen } from 'lucide-react';
import axios from 'axios';

const Sidebar = ({ isOpen, setIsOpen, currentSessionId, setCurrentSessionId }) => {
    const [sessions, setSessions] = React.useState([]);

    // Mock sessions for now
    React.useEffect(() => {
        // In real app, fetch from backend
        setSessions([
            { id: 1, title: 'New Conversation' }
        ]);
    }, []);

    const createNewSession = () => {
        // Logic to create new session
        console.log("Create new session");
    };

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 md:hidden"
                    onClick={() => setIsOpen(false)}
                />
            )}

            {/* Sidebar Panel */}
            <div className={`
        fixed inset-y-0 left-0 z-40 w-72 bg-white border-r border-slate-200 transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        md:relative md:translate-x-0
        ${!isOpen && 'md:hidden'}
      `}>
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="p-4 border-b border-slate-100 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <img src="/vite.svg" alt="Logo" className="w-8 h-8" />
                            {/* Note: Ensure logo exists or use placeholder */}
                            <h1 className="font-bold text-lg text-slate-800">Gazette Chatbot</h1>
                        </div>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="md:hidden p-1 hover:bg-slate-100 rounded"
                        >
                            <X size={20} />
                        </button>
                    </div>

                    {/* New Chat Button */}
                    <div className="p-4">
                        <button
                            onClick={createNewSession}
                            className="w-full flex items-center gap-2 justify-center bg-primary text-white py-2.5 px-4 rounded-lg hover:bg-blue-700 transition-colors shadow-sm font-medium"
                        >
                            <Plus size={20} />
                            New Chat
                        </button>
                    </div>

                    {/* Session List */}
                    <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
                        <div className="text-xs font-semibold text-slate-400 px-3 mb-2 uppercase tracking-wider">Recents</div>
                        {sessions.map(session => (
                            <button
                                key={session.id}
                                onClick={() => setCurrentSessionId(session.id)}
                                className={`w-full flex items-center gap-3 p-3 rounded-md text-left transition-colors group
                  ${currentSessionId === session.id ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}
                `}
                            >
                                <MessageSquare size={18} />
                                <span className="truncate text-sm font-medium">{session.title}</span>
                            </button>
                        ))}
                    </div>

                    {/* Footer / Settings */}
                    <div className="p-4 border-t border-slate-100 bg-slate-50">
                        <button className="flex items-center gap-3 w-full p-2 text-slate-600 hover:text-slate-900 transition-colors">
                            <Settings size={18} />
                            <span className="text-sm font-medium">Settings</span>
                        </button>
                        <div className="mt-3 flex items-center gap-3 w-full p-2 text-slate-500 text-xs">
                            <BookOpen size={14} />
                            <span>Powered by Groq Llama 3</span>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Sidebar;
