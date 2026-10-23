export default function ChatMessage({ item }) {
  const handlePreview = () => {
    if (!item.file_name) return;
    window.open(`/preview/${encodeURIComponent(item.file_name)}`, "_blank");
  };

  return (
    <>
      <div className="message user">
        <div className="message-bubble">{item.question}</div>
      </div>
      <div className="message assistant">
        <div className="message-bubble">{item.answer}</div>
        {item.sources && item.sources.length > 0 && (
          <div className="source-actions">
            <small>
              Sources:{" "}
              {[...new Set(item.sources.map((s) => `${s.file_name} (v${s.version})`))].join(", ")}
            </small>
          </div>
        )}
        {item.file_name && (
          <div className="source-actions">
            <button className="preview-btn" onClick={handlePreview}>
              Preview
            </button>
          </div>
        )}
      </div>
    </>
  );
}
