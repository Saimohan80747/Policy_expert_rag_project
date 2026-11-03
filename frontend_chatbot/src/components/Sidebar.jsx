import FileUpload from "./FileUpload";
import SessionList from "./SessionList";

export default function Sidebar(props) {
  const { refreshKey, onSessionDeleted } = props;
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Document</h2>
      </div>
      <div className="sidebar-section">
        <FileUpload />
      </div>
      <div className="sidebar-section" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", borderBottom: "none" }}>
        <h3>Chat History</h3>
        <SessionList refreshKey={refreshKey} onSessionDeleted={onSessionDeleted} />
      </div>
    </aside>
  );
}
