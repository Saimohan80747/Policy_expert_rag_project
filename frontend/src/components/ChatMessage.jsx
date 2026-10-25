const detailsStyle = { marginTop: 6, fontSize: "0.85em" };
const codeStyle = {
  background: "rgba(127,127,127,0.12)",
  padding: "8px 10px",
  borderRadius: 6,
  overflowX: "auto",
  whiteSpace: "pre-wrap",
};

export default function ChatMessage({ item }) {
  const handlePreview = () => {
    if (!item.file_name) return;
    window.open(`/preview/${encodeURIComponent(item.file_name)}`, "_blank");
  };

  const all = item.sources || [];
  const docs = all.filter((s) => s.type !== "graph");
  const graph = all.find((s) => s.type === "graph");

  return (
    <>
      <div className="message user">
        <div className="message-bubble">{item.question}</div>
      </div>
      <div className="message assistant">
        <div className="message-bubble">{item.answer}</div>
        {docs.length > 0 && (
          <div className="source-actions">
            <small>
              Sources:{" "}
              {[...new Set(docs.map((s) => `${s.file_name} (v${s.version})`))].join(", ")}
            </small>
          </div>
        )}
        {graph && (
          <details className="source-actions" style={detailsStyle}>
            <summary>
              From the knowledge graph ({graph.rows} result{graph.rows === 1 ? "" : "s"}
              {graph.as_of ? `, as of ${graph.as_of}` : ""})
            </summary>
            {graph.entities?.length > 0 && (
              <p>
                Entities:{" "}
                {graph.entities.map((e) => `${e.name} (${e.label})`).join(", ")}
              </p>
            )}
            {graph.facts?.map((fact, i) => (
              <div key={i}>
                [G{i + 1}] {fact}
              </div>
            ))}
            {graph.cypher && <pre style={codeStyle}>{graph.cypher}</pre>}
          </details>
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
