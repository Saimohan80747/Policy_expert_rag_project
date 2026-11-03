import { useState } from "react";
import { uploadFile } from "../api";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setStatus({ type: "warning", message: "Select a file first." });
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      await uploadFile(file);
      setStatus({ type: "success", message: "Document uploaded successfully." });
      setFile(null);
    } catch (err) {
      setStatus({ type: "error", message: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="file-upload">
      <input
        type="file"
        accept=".pdf,.docx,.txt,.csv,.xlsx"
        onChange={(e) => {
          setFile(e.target.files[0] || null);
          setStatus(null);
        }}
      />
      <button
        className="btn btn-primary"
        onClick={handleUpload}
        disabled={loading}
        style={{ width: "100%" }}
      >
        {loading ? "Uploading..." : "Upload Document"}
      </button>
      {status && (
        <div className={`alert alert-${status.type}`}>{status.message}</div>
      )}
    </div>
  );
}
