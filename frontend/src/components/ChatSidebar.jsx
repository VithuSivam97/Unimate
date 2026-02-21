export default function ChatSidebar() {
    return (
        <div className="w-64 bg-white border-r border-gray-300 flex flex-col">
            <div className="p-4 font-bold text-lg border-b border-gray-200">Chats</div>
            <div className="flex-1 overflow-y-auto">
                <div className="p-4 hover:bg-gray-100 cursor-pointer">Chat 1</div>
                <div className="p-4 hover:bg-gray-100 cursor-pointer">Chat 2</div>
            </div>
            <div className="p-4 border-t border-gray-200 text-sm text-gray-500">
                GPT Clone
            </div>
        </div>
    );
}
