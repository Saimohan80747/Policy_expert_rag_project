import { useState } from "react";
import { uploadFile } from "../api";

function describeResult(result) {
  if (result.ingested_files) {
    const ok = result.ingested_files.length;
    const dup = result.duplicate_files.length;
    const bad = result.failed_files.length;
    return {
      type: bad ? "warning" : "success",
      message: `ZIP processed: ${ok} ingested, ${dup} duplicate, ${bad} failed.`,
    };
  }
  if (result.status === "ingested") {
    return {
      type: "success",
      message: `Ingested ${result.file_name} (version ${result.version}, ${result.chunks} chunks).`,
    };
  }
  if (result.status === "duplicate") {
    return { type: "warning", message: `Already ingested as ${result.duplicate_of}.` };
  }
  return { type: "error", message: result.error || "Ingestion failed." };
}

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
      const result = await uploadFile(file);
      setStatus(describeResult(result));
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
