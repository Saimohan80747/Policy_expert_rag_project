import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import PreviewPage from "./pages/PreviewPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/session/:sessionId" element={<HomePage />} />
      <Route path="/preview/:fileName" element={<PreviewPage />} />
    </Routes>
  );
}
