import { useState } from "react";
import { sendPrompt } from "../api";

export default function ChatInput({ sessionId, onMessageSent }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);

    try {
      const result = await sendPrompt(trimmed, sessionId || "");
      onMessageSent(trimmed, result);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="chat-input-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        disabled={loading}
      />
      <button className="btn btn-primary" type="submit" disabled={loading || !question.trim()}>
        {loading ? "Sending..." : "Send"}
      </button>
      {error && (
        <div className="alert alert-error" style={{ position: "absolute", bottom: "70px", left: "24px", right: "24px" }}>
          {error}
        </div>
      )}
    </form>
  );
}
