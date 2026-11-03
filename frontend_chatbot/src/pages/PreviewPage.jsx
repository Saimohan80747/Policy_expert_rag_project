import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { downloadFile, getPreview } from "../api";

export default function PreviewPage() {
  const { fileName: encodedFileName } = useParams();
  const fileName = decodeURIComponent(encodedFileName || "");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!fileName) {
      setError("No file specified.");
      setLoading(false);
      return;
    }

    getPreview(fileName)
      .then((data) => setText(data.text || ""))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [fileName]);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const blob = await downloadFile(fileName);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="preview-page">
      <header className="preview-header">
        <h1>{fileName || "Preview"}</h1>
        <button
          className="btn btn-primary"
          onClick={handleDownload}
          disabled={downloading || loading || !!error}
        >
          {downloading ? "Downloading..." : "Download"}
        </button>
      </header>
      <div className="preview-content">
        {loading && <p className="preview-loading">Loading preview...</p>}
        {error && <p className="preview-error">{error}</p>}
        {!loading && !error && <pre className="preview-text">{text}</pre>}
      </div>
    </div>
  );
}
