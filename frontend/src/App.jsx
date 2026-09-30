import { useState, useRef, useCallback, useEffect } from "react";

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
          ? <svg className="w-3 h-3 text-emerald-500 dark:text-emerald-400" fill="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /></svg>
          : <svg className="w-3 h-3 text-slate-300 dark:text-slate-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /></svg>
        }
        <span className="text-xs text-slate-600 dark:text-slate-400">{DESK_LABELS[desk] || desk}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className={`text-xs font-semibold ${data.occupied ? "text-emerald-600 dark:text-emerald-400" : "text-slate-400 dark:text-slate-500"}`}>
          {data.occupied ? "Occ" : "Empty"}
        </span>
        <span className="text-xs font-mono text-slate-400 dark:text-slate-500">{(data.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

function ResultCard({ preview, result, error, index, onRetry, isRetrying }) {
  const occupiedCount = result ? Object.values(result.desks || {}).filter((d) => d.occupied).length : 0;
  const totalCount = result ? Object.keys(result.desks || {}).length : 0;

  return (
    <div className="w-full flex-shrink-0 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className="relative">
        <img src={preview} alt={"Result " + (index + 1)} className="w-full h-28 object-cover" />
        {error && (
          <div className="absolute inset-0 bg-red-50/80 dark:bg-red-900/80 flex flex-col items-center justify-center gap-1 p-2">
            <span className="text-xs text-red-500 dark:text-red-300 font-medium text-center">{error}</span>
            <button
              onClick={() => onRetry(index)}
              disabled={isRetrying}
              className="text-xs px-2 py-1 bg-red-100 dark:bg-red-800 hover:bg-red-200 dark:hover:bg-red-700 text-red-600 dark:text-red-300 rounded transition-colors disabled:opacity-50 cursor-pointer"
            >
              {isRetrying ? "Retrying..." : "Retry"}
            </button>
          </div>
        )}
        {result && (
          <div className="absolute top-1.5 right-1.5">
            <span className="text-xs font-mono bg-white/90 dark:bg-slate-700/90 text-slate-600 dark:text-slate-300 px-1.5 py-0.5 rounded-md">
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
        <div className="divide-y divide-slate-100 dark:divide-slate-700">
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

function SunIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
    </svg>
  );
}

// NOTE: For batches > 100 images, consider react-window virtualization
// to avoid DOM performance issues. Replace the results grid with:
// <FixedSizeList> or <VariableSizeList> from react-window

export default function App() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [retryingIndex, setRetryingIndex] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cape-dark-mode");
      if (saved !== null) return saved === "true";
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const blobUrlsRef = useRef(new Set());

  // Apply dark mode class to html element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("cape-dark-mode", String(darkMode));
  }, [darkMode]);

  const revokeAllBlobUrls = useCallback(() => {
    blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    blobUrlsRef.current.clear();
  }, []);

  function handleFileChange(e) {
    revokeAllBlobUrls();
    const selected = Array.from(e.target.files).filter((f) => f.type.startsWith("image/"));
    if (!selected.length) return;
    setFiles(selected);
    setResults([]);
    setProgress({ current: 0, total: 0 });
  }

  function handleDrop(e) {
    e.preventDefault();
    revokeAllBlobUrls();
    const dropped = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (!dropped.length) return;
    setFiles(dropped);
    setResults([]);
    setProgress({ current: 0, total: 0 });
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  }

  function handleReset(e) {
    e.stopPropagation();
    revokeAllBlobUrls();
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
      await processFile(i);
      setProgress((p) => ({ ...p, current: i + 1 }));
      if (i < files.length - 1) {
        await sleep(INTERVAL_MS);
      }
    }

    setRunning(false);
  }

  async function processFile(i) {
    const file = files[i];
    const blobUrl = URL.createObjectURL(file);
    blobUrlsRef.current.add(blobUrl);

    setResults((prev) => {
      const next = [...prev];
      next[i] = { preview: blobUrl, result: null, error: null };
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
          next[i] = { preview: blobUrl, result: null, error: data.detail || ("Error " + res.status) };
          return next;
        });
      } else {
        setResults((prev) => {
          const next = [...prev];
          next[i] = { preview: blobUrl, result: data, error: null };
          return next;
        });
      }
    } catch (err) {
      setResults((prev) => {
        const next = [...prev];
        next[i] = { preview: blobUrl, result: null, error: err.message || "Failed" };
        return next;
      });
    }
  }

  async function handleRetry(index) {
    if (retryingIndex !== null) return;
    setRetryingIndex(index);
    const oldUrl = results[index]?.preview;
    if (oldUrl) {
      blobUrlsRef.current.delete(oldUrl);
      URL.revokeObjectURL(oldUrl);
    }
    setResults((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], result: null, error: null };
      return next;
    });
    await processFile(index);
    setRetryingIndex(null);
  }

  const photoLabel = files.length === 1 ? "1 photo" : (files.length + " photos");
  const successCount = results.filter((r) => r.result).length;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-start justify-center px-4 py-10">
      <div className="w-full max-w-4xl space-y-6">

        <div className="flex items-center justify-between">
          <div className="text-center space-y-1 flex-1">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 tracking-tight">Batch Detection</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">Select multiple photos — auto-detects every {INTERVAL_MS / 1000}s</p>
          </div>
          <button
            onClick={() => setDarkMode((d) => !d)}
            aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
            className="p-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors cursor-pointer"
          >
            {darkMode ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>

        <div
          ref={dropZoneRef}
          role="button"
          tabIndex={0}
          aria-label="Upload photos: drag and drop or click to browse"
          aria-describedby="dropzone-hint"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onKeyDown={handleKeyDown}
          className={"w-full border-2 border-dashed rounded-xl p-8 text-center cursor-pointer select-none transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-slate-900 " + (files.length ? "border-emerald-300 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-900/20" : "border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-50 dark:hover:bg-slate-700/50")}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileChange}
            className="hidden"
            aria-hidden="true"
          />
          <div className="flex flex-col items-center gap-2">
            <svg className="w-8 h-8 text-slate-400 dark:text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            {files.length > 0 ? (
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">{photoLabel} selected</p>
            ) : (
              <div>
                <p className="text-sm font-medium text-slate-600 dark:text-slate-300">Drag &amp; drop photos, or <span className="text-blue-500 dark:text-blue-400">click to browse</span></p>
                <p id="dropzone-hint" className="text-xs text-slate-400 dark:text-slate-500">JPG, PNG, WEBP — multiple files supported</p>
              </div>
            )}
          </div>
        </div>

        {files.length > 0 && (
          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleReset}
              className="text-xs text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors cursor-pointer"
            >
              Clear selection
            </button>
          </div>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={runBatch}
            disabled={!files.length || running}
            className={"flex-1 py-3 rounded-xl text-sm font-semibold transition-all duration-150 cursor-pointer disabled:cursor-not-allowed " + (!files.length || running ? "bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500" : "bg-blue-500 dark:bg-blue-600 text-white hover:bg-blue-600 dark:hover:bg-blue-500 active:bg-blue-700 dark:active:bg-blue-400")}
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
              <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Results</p>
              <p className="text-xs text-slate-400 dark:text-slate-500">{successCount}/{results.length} successful</p>
            </div>
            <div
              role="list"
              aria-label="Detection results"
              className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2"
            >
              {results.map((r, i) => (
                <div key={i} role="listitem">
                  <ResultCard
                    preview={r.preview}
                    result={r.result}
                    error={r.error}
                    index={i}
                    onRetry={handleRetry}
                    isRetrying={retryingIndex === i}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
