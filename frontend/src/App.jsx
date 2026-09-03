import { useState, useRef } from "react";

const API_URL = "/detect";

const DESK_LABELS = {
  desk_a: "Desk A",
  desk_b: "Desk B",
  desk_c: "Desk C",
  desk_d: "Desk D",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  function handleFile(f) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  function handleFileChange(e) {
    handleFile(e.target.files[0]);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files[0]);
  }

  function handleDragOver(e) {
    e.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave() {
    setDragActive(false);
  }

  function handleReset(e) {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsDataURL(file);
    });

    // strip "data:image/...;base64," prefix
    const raw = base64.split(",")[1];
    if (!raw) {
      setError("Failed to encode image");
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: raw, angle: null }),
      });

      const errData = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      setResult(errData);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center px-4 py-16">
      <div className="w-full max-w-lg space-y-8">

        {/* Header */}
        <div className="text-center space-y-1">
          <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
            Desk Detection
          </h1>
          <p className="text-sm text-slate-500">
            Upload an image — angle is auto-detected
          </p>
        </div>

        {/* Upload zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            w-full border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
            transition-colors duration-150 select-none
            ${dragActive
              ? "border-blue-400 bg-blue-50"
              : file
                ? "border-emerald-300 bg-emerald-50"
                : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50"
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          {preview ? (
            <div className="space-y-3">
              <img
                src={preview}
                alt="Preview"
                className="max-h-52 mx-auto rounded-lg object-contain"
              />
              <button
                type="button"
                onClick={handleReset}
                className="text-xs text-slate-400 hover:text-red-500 transition-colors cursor-pointer"
              >
                Remove image
              </button>
            </div>
          ) : (
            <div className="space-y-3 pointer-events-none">
              <div className="flex justify-center">
                <svg
                  className={`w-9 h-9 ${dragActive ? "text-blue-400" : "text-slate-400"}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
              </div>
              <p className="text-sm font-medium text-slate-600">
                Drag & drop an image, or{" "}
                <span className="text-blue-500">click to browse</span>
              </p>
              <p className="text-xs text-slate-400">JPG, PNG, WEBP</p>
            </div>
          )}
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className={`
            w-full py-3 rounded-xl text-sm font-semibold transition-all duration-150 cursor-pointer
            ${!file || loading
              ? "bg-slate-100 text-slate-400 cursor-not-allowed"
              : "bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700"
            }
          `}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              Detecting...
            </span>
          ) : (
            "Run Detection"
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-xl">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="w-full space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                {result.angle?.replace("angle_", "Angle ") || "Results"}
              </p>
              <p className="text-xs text-slate-400">
                {Object.values(result.desks || {}).filter((d) => d.occupied).length} /{" "}
                {Object.keys(result.desks || {}).length} occupied
              </p>
            </div>

            <div className="space-y-1">
              {Object.entries(result.desks || {}).map(([key, data]) => (
                <div
                  key={key}
                  className="flex items-center justify-between px-4 py-3 bg-white rounded-xl border border-slate-200"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-base leading-none">
                      {data.occupied
                        ? <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /></svg>
                        : <svg className="w-4 h-4 text-slate-300" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /></svg>
                      }
                    </span>
                    <span className="text-sm font-medium text-slate-800">
                      {DESK_LABELS[key] || key.replace("desk_", "Desk ").toUpperCase()}
                    </span>
                    <span className="text-sm text-slate-400">— {data.person || "—"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-semibold ${data.occupied ? "text-emerald-600" : "text-slate-400"}`}>
                      {data.occupied ? "Occupied" : "Empty"}
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      {(data.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
