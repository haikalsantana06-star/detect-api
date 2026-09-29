import { useState, useRef } from "react";

const API_URL = "/detect";
const INTERVAL_MS = 2000;

const DESK_LABELS = {
  desk_a: "Desk A",
  desk_b: "Desk B",
  desk_c: "Desk C",
  desk_d: "Desk D",
};

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function DeskRow({ desk, data }) {
  return (
    <div className="flex items-center justify-between px-2 py-1">
      <div className="flex items-center gap-1.5">
        {data.occupied
          ? <svg className="w-3 h-3 text-emerald-500" fill="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /></svg>
          : <svg className="w-3 h-3 text-slate-300" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /></svg>
        }
        <span className="text-xs text-slate-600">{DESK_LABELS[desk] || desk}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className={`text-xs font-semibold ${data.occupied ? "text-emerald-600" : "text-slate-400"}`}>
          {data.occupied ? "Occ" : "Empty"}
        </span>
        <span className="text-xs font-mono text-slate-400">{(data.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

function ResultCard({ preview, result, error, index }) {
  const occupiedCount = result ? Object.values(result.desks || {}).filter((d) => d.occupied).length : 0;
  const totalCount = result ? Object.keys(result.desks || {}).length : 0;

  return (
    <div className="w-40 flex-shrink-0 bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="relative">
        <img src={preview} alt={"Result " + (index + 1)} className="w-full h-28 object-cover" />
        {error && (
          <div className="absolute inset-0 bg-red-50/80 flex items-center justify-center">
            <span className="text-xs text-red-500 font-medium text-center px-2">{error}</span>
          </div>
        )}
        {result && (
          <div className="absolute top-1.5 right-1.5">
            <span className="text-xs font-mono bg-white/90 text-slate-600 px-1.5 py-0.5 rounded-md">
              {result.angle ? result.angle.replace("angle_", "") : "?"}
            </span>
          </div>
        )}
        {result && (
          <div className="absolute bottom-1.5 left-1.5">
            <span className={"text-xs font-semibold px-1.5 py-0.5 rounded-md " + (occupiedCount > 0 ? "bg-emerald-500/90 text-white" : "bg-slate-500/80 text-white")}>
              {occupiedCount}/{totalCount}
            </span>
          </div>
        )}
      </div>
      {result && (
        <div className="divide-y divide-slate-100">
          {Object.entries(result.desks || {}).map(([desk, data]) => (
            <DeskRow key={desk} desk={desk} data={data} />
          ))}
        </div>
      )}
    </div>
  );
}

async function encodeFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const fileInputRef = useRef(null);

  function handleFileChange(e) {
    const selected = Array.from(e.target.files).filter((f) => f.type.startsWith("image/"));
    if (!selected.length) return;
    setFiles(selected);
    setResults([]);
    setProgress({ current: 0, total: 0 });
  }

  function handleDrop(e) {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (!dropped.length) return;
    setFiles(dropped);
    setResults([]);
    setProgress({ current: 0, total: 0 });
  }

  function handleReset(e) {
    e.stopPropagation();
    setFiles([]);
    setResults([]);
    setProgress({ current: 0, total: 0 });
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function runBatch() {
    if (!files.length) return;
    setRunning(true);
    setResults([]);
    setProgress({ current: 0, total: files.length });

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const preview = URL.createObjectURL(file);

      setResults((prev) => {
        const next = [...prev];
        next[i] = { preview, result: null, error: null };
        return next;
      });

      try {
        const raw = await encodeFile(file);
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: raw, angle: null }),
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          setResults((prev) => {
            const next = [...prev];
            next[i] = { preview, result: null, error: data.detail || ("Error " + res.status) };
            return next;
          });
        } else {
          setResults((prev) => {
            const next = [...prev];
            next[i] = { preview, result: data, error: null };
            return next;
          });
        }
      } catch (err) {
        setResults((prev) => {
          const next = [...prev];
          next[i] = { preview, result: null, error: err.message || "Failed" };
          return next;
        });
      }

      setProgress((p) => ({ ...p, current: i + 1 }));

      if (i < files.length - 1) {
        await sleep(INTERVAL_MS);
      }
    }

    setRunning(false);
  }

  const photoLabel = files.length === 1 ? "1 photo" : (files.length + " photos");
  const successCount = results.filter((r) => r.result).length;

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center px-4 py-10">
      <div className="w-full max-w-4xl space-y-6">

        <div className="text-center space-y-1">
          <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Batch Detection</h1>
          <p className="text-sm text-slate-500">Select multiple photos — auto-detects every {INTERVAL_MS / 1000}s</p>
        </div>

        <div
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className={"w-full border-2 border-dashed rounded-xl p-8 text-center cursor-pointer select-none transition-colors duration-150 " + (files.length ? "border-emerald-300 bg-emerald-50" : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50")}
        >
          <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleFileChange} className="hidden" />
          <div className="flex flex-col items-center gap-2">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            {files.length > 0 ? (
              <div>
                <p className="text-sm font-medium text-slate-700">{photoLabel} selected</p>
                <button type="button" onClick={handleReset} className="text-xs text-slate-400 hover:text-red-500 transition-colors cursor-pointer mt-1">Clear selection</button>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium text-slate-600">Drag &amp; drop photos, or <span className="text-blue-500">click to browse</span></p>
                <p className="text-xs text-slate-400">JPG, PNG, WEBP — multiple files supported</p>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={runBatch}
            disabled={!files.length || running}
            className={"flex-1 py-3 rounded-xl text-sm font-semibold transition-all duration-150 cursor-pointer " + (!files.length || running ? "bg-slate-100 text-slate-400 cursor-not-allowed" : "bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700")}
          >
            {running ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                </svg>
                Detecting {progress.current}/{progress.total}...
              </span>
            ) : "Start Detection"}
          </button>
        </div>

        {results.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Results</p>
              <p className="text-xs text-slate-400">{successCount}/{results.length} successful</p>
            </div>
            <div className="flex flex-row-reverse flex-wrap gap-2">
              {results.map((r, i) => (
                <ResultCard key={i} preview={r.preview} result={r.result} error={r.error} index={i} />
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
