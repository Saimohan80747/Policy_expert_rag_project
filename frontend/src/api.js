export const BASE_URL = "http://127.0.0.1:8000";

async function apiGet(path) {
  const response = await fetch(`${BASE_URL}${path}`);
  return response;
}

async function apiPost(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    ...options,
  });
  return response;
}

async function apiDelete(path) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "DELETE",
  });
  return response;
}

export async function getHistory() {
  const response = await apiGet("/history");
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  return data.sessions || [];
}

export async function getSessionMessages(sessionId) {
  const response = await apiGet(`/sessions/${sessionId}`);
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  return data.session || [];
}

export async function sendPrompt(question, sessionId) {
  const response = await apiPost("/prompt", {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiPost("/pipeline", { body: formData });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function deleteSession(sessionId) {
  const response = await apiDelete(`/sessions/${sessionId}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getPreview(fileName) {
  const response = await apiGet(`/preview/${encodeURIComponent(fileName)}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export function getDownloadUrl(fileName) {
  return `${BASE_URL}/download/${encodeURIComponent(fileName)}`;
}

export async function downloadFile(fileName) {
  const response = await fetch(getDownloadUrl(fileName));
  if (!response.ok) throw new Error("Download failed");
  return response.blob();
}
