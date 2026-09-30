const API_URL = "/stats";

const PERSON_OPTIONS = [
  { value: "all", label: "All People" },
  { value: "Asep", label: "Asep" },
  { value: "Budi", label: "Budi" },
  { value: "Saep", label: "Saep" },
  { value: "naisya", label: "Naisya" },
];

const _DESK_LABELS = {
  desk_a: "Desk A",
  desk_b: "Desk B",
  desk_c: "Desk C",
  desk_d: "Desk D",
};

function formatDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString("id-ID", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function formatMinutesToHours(minutes) {
  if (minutes === null || minutes === undefined || minutes === 0) {
    return "0h 0m";
  }
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function formatTime(timeStr) {
  if (!timeStr || timeStr === "N/A" || timeStr === "-") {
    return "—";
  }
  return timeStr;
}

function StatusBadge({ present }) {
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
      present
        ? "bg-emerald-100 text-emerald-700"
        : "bg-red-100 text-red-700"
    }`}>
      {present ? (
        <>
          <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          Hadir
        </>
      ) : (
        <>
          <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
          Tidak Hadir
        </>
      )}
    </span>
  );
}

function KPICard({ icon, label, value, unit, subValue, colorClass }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className={`p-2 rounded-lg ${colorClass}`}>
          {icon}
        </div>
      </div>
      <div className="mt-4">
        <p className="text-sm text-slate-500 font-medium">{label}</p>
        <p className="text-2xl font-bold text-slate-900 mt-1">{value}</p>
        {unit && <p className="text-sm text-slate-400 mt-0.5">{unit}</p>}
        {subValue && <p className="text-xs text-slate-400 mt-1">{subValue}</p>}
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <svg className="w-10 h-10 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
      </svg>
      <p className="mt-4 text-slate-500">Memuat data...</p>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <svg className="w-12 h-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
      </svg>
      <p className="mt-4 text-slate-600 font-medium">Gagal memuat data</p>
      <p className="text-sm text-slate-400 mt-1">{message}</p>
      <button
        onClick={onRetry}
        className="mt-4 px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-lg hover:bg-blue-600 transition-colors cursor-pointer"
      >
        Coba Lagi
      </button>
    </div>
  );
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <svg className="w-12 h-12 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-2.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
      </svg>
      <p className="mt-4 text-slate-500 font-medium">{message}</p>
    </div>
  );
}

export default function StatsDashboard() {
  const today = new Date().toISOString().split("T")[0];

  const [selectedPerson, setSelectedPerson] = useState("all");
  const [selectedDate, setSelectedDate] = useState(today);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStats = useCallback(async () => {
    if (selectedPerson === "all") return;

    setLoading(true);
    setError(null);
    setStats(null);

    try {
      const res = await fetch(`${API_URL}/${selectedPerson}?date=${selectedDate}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      setStats(data);
    } catch (err) {
      setError(err.message || "Failed to fetch stats");
    } finally {
      setLoading(false);
    }
  }, [selectedPerson, selectedDate]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const indicators = stats?.indicators || {};

  const tingkatKehadiran = indicators["tingkat_kehadiran"];
  const ketepatanDatang = indicators["ketepatan_datang"];
  const lamaBekerja = indicators["lama_bekerja"];
  const waktuProduktif = indicators["waktu_produktif"];
  const waktuTidakProduktif = indicators["waktu_tidak_produktif"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-1">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Dashboard Statistik</h1>
        <p className="text-sm text-slate-500">5 Indikator Manajemen Kehadiran</p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Orang</label>
            <select
              value={selectedPerson}
              onChange={(e) => setSelectedPerson(e.target.value)}
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
            >
              {PERSON_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Tanggal</label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Date Display */}
      {selectedDate && (
        <p className="text-sm text-slate-500 text-center">{formatDate(selectedDate)}</p>
      )}

      {/* Content */}
      {selectedPerson === "all" ? (
        <EmptyState message="Pilih orang untuk melihat statistik" />
      ) : loading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState message={error} onRetry={fetchStats} />
      ) : stats ? (
        <div className="space-y-6">
          {/* Person & Work Hours Info */}
          <div className="bg-slate-100 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-lg font-semibold text-slate-800 capitalize">{stats.person}</p>
                <p className="text-sm text-slate-500 mt-0.5">
                  Jam kerja: {stats.work_hours.start} - {stats.work_hours.end}
                </p>
              </div>
              <StatusBadge present={tingkatKehadiran?.value === true} />
            </div>
          </div>

          {/* KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Ketepatan Datang */}
            <KPICard
              icon={
                <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2m6-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              }
              label="Ketepatan Datang"
              value={formatTime(ketepatanDatang?.value)}
              colorClass="bg-blue-100"
            />

            {/* Lama Bekerja */}
            <KPICard
              icon={
                <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
                </svg>
              }
              label="Lama Bekerja"
              value={formatMinutesToHours(lamaBekerja?.value)}
              unit={`Target: ${formatMinutesToHours(stats.work_hours.total_minutes)}`}
              colorClass="bg-emerald-100"
            />

            {/* Waktu Produktif */}
            <KPICard
              icon={
                <svg className="w-5 h-5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              }
              label="Waktu Produktif"
              value={formatMinutesToHours(waktuProduktif?.value)}
              colorClass="bg-teal-100"
            />

            {/* Waktu Tidak Produktif */}
            <KPICard
              icon={
                <svg className="w-5 h-5 text-orange-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              }
              label="Waktu Tidak Produktif"
              value={formatMinutesToHours(waktuTidakProduktif?.value)}
              colorClass="bg-orange-100"
            />

            {/* Tingkat Kehadiran (Full Width) */}
            <KPICard
              icon={
                <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              }
              label="Tingkat Kehadiran"
              value={tingkatKehadiran?.value === true ? "Hadir" : "Tidak Hadir"}
              colorClass="bg-purple-100"
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}
