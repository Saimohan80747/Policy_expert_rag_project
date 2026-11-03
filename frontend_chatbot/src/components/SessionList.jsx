import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { deleteSession, getHistory } from "../api";

export default function SessionList({ refreshKey, onSessionDeleted }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams();

  const loadSessions = async () => {
    try {
      const data = await getHistory();
      setSessions(data);
    } catch {
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, [refreshKey]);

  const handleSelect = (id) => {
    navigate(`/session/${id}`);
  };

  const handleDelete = async (id) => {
    try {
      await deleteSession(id);
      if (activeSessionId === id) {
        navigate("/");
      }
      onSessionDeleted?.();
      loadSessions();
    } catch {
      /* ignore */
    }
  };

  if (loading) {
    return <p className="empty-text">Loading...</p>;
  }

  if (sessions.length === 0) {
    return <p className="empty-text">No previous chats.</p>;
  }

  return (
    <div className="session-list">
      {sessions.map((session) => (
        <div className="session-item" key={session.id}>
          <button
            className={`session-btn ${activeSessionId === session.id ? "active" : ""}`}
            onClick={() => handleSelect(session.id)}
            title={session.title || "Untitled"}
          >
            {session.title || "Untitled"}
          </button>
          <button
            className="delete-btn"
            onClick={() => handleDelete(session.id)}
            title="Delete session"
          >
            🗑
          </button>
        </div>
      ))}
    </div>
  );
}
