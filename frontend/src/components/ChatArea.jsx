import ChatMessage from "./ChatMessage";

export default function ChatArea({ messages }) {
  if (messages.length === 0) {
    return (
      <div className="chat-area">
        <div className="chat-empty">Ask a question about your uploaded documents.</div>
      </div>
    );
  }

  return (
    <div className="chat-area">
      {messages.map((item, index) => (
        <ChatMessage key={item.id || index} item={item} />
      ))}
    </div>
  );
}
