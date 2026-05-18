// App.jsx — Root component
// Upload ảnh + nhập keywords → gọi API → hiển thị CanvasComponent

import { useState, useRef } from "react";
import CanvasComponent from "./CanvasComponent";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8009";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [keywords, setKeywords] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [canvasData, setCanvasData] = useState(null); // kết quả từ API
  const fileInputRef = useRef(null);

  // ── Chọn file ──
  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setCanvasData(null);
    setError(null);
  };

  // ── Drag & drop ──
  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setCanvasData(null);
      setError(null);
    }
  };

  // ── Submit ──
  const handleSubmit = async () => {
    if (!file) { setError("Vui lòng chọn ảnh."); return; }
    if (!keywords.trim()) { setError("Vui lòng nhập ít nhất một từ khóa."); return; }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("keywords", keywords);

    try {
      const res = await fetch(`${API_BASE}/api/process-image`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Server error");
      }

      const data = await res.json();
      setCanvasData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Nếu đã có kết quả: hiển thị Canvas editor ──
  if (canvasData) {
    return (
      <div className="flex flex-col h-screen bg-gray-950">
        {/* Toolbar */}
        <header className="flex items-center gap-4 px-6 py-3 bg-gray-900 border-b border-gray-800">
          <span className="text-white font-semibold text-lg tracking-tight">
            ✦ AI Magic Canvas
          </span>
          <div className="flex-1" />
          <span className="text-gray-400 text-sm">
            {canvasData.layers.length} đối tượng •{" "}
            {canvasData.original_width} × {canvasData.original_height}px
          </span>
          <button
            onClick={() => setCanvasData(null)}
            className="px-4 py-1.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition"
          >
            ← Quay lại
          </button>
        </header>

        {/* Sidebar + Canvas */}
        <div className="flex flex-1 overflow-hidden">
          {/* Layer panel */}
          <aside className="w-56 bg-gray-900 border-r border-gray-800 p-3 flex flex-col gap-2 overflow-y-auto">
            <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Layers</p>
            <div className="flex items-center gap-2 px-2 py-2 rounded-lg bg-gray-800 text-sm text-gray-300">
              <span className="w-3 h-3 rounded-sm bg-gray-500 inline-block" />
              Background (LaMa)
            </div>
            {canvasData.layers.map((layer, i) => (
              <div
                key={i}
                className="flex items-center gap-2 px-2 py-2 rounded-lg bg-gray-800 text-sm text-gray-300"
              >
                <span className="w-3 h-3 rounded-sm bg-indigo-500 inline-block" />
                {layer.keyword}
              </div>
            ))}
          </aside>

          {/* Canvas area */}
          <main className="flex-1 flex items-center justify-center bg-gray-950 overflow-auto min-w-0 min-h-0 p-4">
            <CanvasComponent data={canvasData} />
          </main>
        </div>
      </div>
    );
  }

  // ── Upload form ──
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-xl">
        <h1 className="text-3xl font-bold text-white mb-1 tracking-tight">
          ✦ AI Magic Canvas
        </h1>
        <p className="text-gray-400 mb-8 text-sm">
          Upload ảnh, nhập từ khóa đối tượng — AI tự động tách layer và inpaint nền.
        </p>

        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-700 hover:border-indigo-500 rounded-2xl p-8 text-center cursor-pointer transition-colors bg-gray-900 mb-4"
        >
          {preview ? (
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 mx-auto rounded-lg object-contain"
            />
          ) : (
            <div className="text-gray-500">
              <div className="text-4xl mb-2">🖼</div>
              <p className="text-sm">Kéo thả ảnh vào đây hoặc click để chọn</p>
              <p className="text-xs mt-1 text-gray-600">JPEG, PNG, WEBP — tối đa 10MB</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {/* Keywords input */}
        <div className="mb-4">
          <label className="block text-gray-300 text-sm mb-1.5">
            Từ khóa đối tượng cần tách
          </label>
          <input
            type="text"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="dog, cat, person, car..."
            className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition"
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          <p className="text-gray-600 text-xs mt-1">
            Ngăn cách bằng dấu phẩy, tối đa 10 từ khóa
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-900/40 border border-red-700 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium text-sm transition flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
              Đang xử lý... (có thể mất 15–60s)
            </>
          ) : (
            "✦ Tách Layer với AI"
          )}
        </button>
      </div>
    </div>
  );
}