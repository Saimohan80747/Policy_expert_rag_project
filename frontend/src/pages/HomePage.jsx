import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getSessionMessages } from "../api";
import Sidebar from "../components/Sidebar";
import ChatArea from "../components/ChatArea";
import ChatInput from "../components/ChatInput";

export default function HomePage() {
  const { sessionId: urlSessionId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(urlSessionId || "");
  const [refreshKey, setRefreshKey] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!urlSessionId) {
      setSessionId("");
      setMessages([]);
      return;
    }

    setSessionId(urlSessionId);
    setLoading(true);

    getSessionMessages(urlSessionId)
      .then((data) => setMessages(data))
      .catch(() => setMessages([]))
      .finally(() => setLoading(false));
  }, [urlSessionId]);

  const handleMessageSent = (question, result) => {
    const newSessionId = result.session_id || sessionId;

    setMessages((prev) => [
      ...prev,
      {
        question,
        answer: result.answer,
        file_name: result.file_name,
        file_url: result.file_url,
        sources: result.sources,
      },
    ]);

    if (newSessionId && newSessionId !== urlSessionId) {
      setSessionId(newSessionId);
      navigate(`/session/${newSessionId}`, { replace: true });
      setRefreshKey((k) => k + 1);
    }
  };

  return (
    <div className="layout">
      <Sidebar
        refreshKey={refreshKey}
        onSessionDeleted={() => {
          setMessages([]);
          setSessionId("");
        }}
      />
      <main className="main">
        <div className="chat-header">
          <h1>RAG Chat Assistant</h1>
        </div>
        {loading ? (
          <div className="chat-area">
            <div className="chat-empty">Loading session...</div>
          </div>
        ) : (
          <ChatArea messages={messages} />
        )}
        <ChatInput sessionId={sessionId} onMessageSent={handleMessageSent} />
      </main>
    </div>
  );
}
