"""Compute 5 management indicators from detection_logs in MySQL."""
import logging
from datetime import date, datetime, time, timedelta, timezone

from config import get_db_connection, get_settings
from stats_schemas import (
    AggregateStats,
    AvailableDates,
    DailyStats,
    DeskStatsResponse,
    Indicator,
    PersonComparison,
    PresenceStatus,
    StatsResponse,
    TrendPoint,
    WorkHours,
)

logger = logging.getLogger(__name__)

# Timezone: Asia/Jakarta = UTC+7
JAKARTA_TZ = timezone(timedelta(hours=7))

# Default work hours (used before config is loaded)
WORK_START = time(7, 0)
WORK_END = time(17, 0)
DETECTION_INTERVAL_MINUTES = 5


def _get_work_start() -> time:
    s = get_settings()
    parts = s.work_start_time.split(":")
    return time(int(parts[0]), int(parts[1]))


def _get_work_end() -> time:
    s = get_settings()
    parts = s.work_end_time.split(":")
    return time(int(parts[0]), int(parts[1]))


def _get_presence_threshold() -> float:
    return get_settings().presence_threshold


def _get_tolerance_minutes() -> int:
    return get_settings().arrival_tolerance_minutes


def _get_interval_minutes() -> int:
    return get_settings().detection_interval_minutes


def _to_jakarta(dt: datetime) -> datetime:
    """Convert naive datetime to Jakarta timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=JAKARTA_TZ)
    return dt.astimezone(JAKARTA_TZ)


def _jakarta_date(dt: datetime) -> date:
    """Get date in Jakarta timezone."""
    return _to_jakarta(dt).date()


def _parse_datetime(dt_val) -> datetime | None:
    """Convert MySQL datetime result to Python datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, str):
        try:
            return datetime.fromisoformat(dt_val)
        except ValueError:
            return datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
    return None


def _default_stats(person: str, target_date: date) -> StatsResponse:
    """Return default stats response with all indicators at zero."""
    work_hours = WorkHours(start="07:00", end="17:00", total_minutes=600)
    return StatsResponse(
        person=person,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators={
            "tingkat_kehadiran": Indicator(
                label="Tingkat Kehadiran",
                value=False,
                description="Hadir jika muncul >= 1x selama jam kerja"
            ),
            "ketepatan_datang": Indicator(
                label="Ketepatan Waktu Datang",
                value="N/A",
                description="Waktu pertama muncul dalam jam kerja (HH:MM)"
            ),
            "lama_bekerja": Indicator(
                label="Lama Berada di Kantor",
                value=0,
                unit="minutes",
                description="Durasi dari first_seen sampai last_seen dalam jam kerja"
            ),
            "waktu_produktif": Indicator(
                label="Waktu Produktif",
                value=0,
                unit="minutes",
                description="Menit occupied selama jam kerja"
            ),
            "waktu_tidak_produktif": Indicator(
                label="Waktu Tidak Produktif",
                value=0,
                unit="minutes",
                description="Menit occupied di luar jam kerja"
            ),
        },
    )


def _build_indicators(
    occupied_in_work_hours: list,
    detections: list,
    all_times: list,
) -> dict[str, Indicator]:
    """Build the 5 indicator objects from computed values."""
    tingkat_kehadiran = len(occupied_in_work_hours) > 0

    ketepatan_datang = "N/A"
    if occupied_in_work_hours:
        first = min(occupied_in_work_hours)
        ketepatan_datang = first.strftime("%H:%M")

    lama_bekerja = 0
    if all_times:
        first_seen_dt = min(all_times)
        last_seen_dt = max(all_times)
        diff_minutes = int((last_seen_dt - first_seen_dt).total_seconds() / 60)
        lama_bekerja = min(diff_minutes, 600)

    productive_minutes = 0
    unproductive_minutes = 0
    for dt, occ in detections:
        if not occ:
            continue
        t = dt.time()
        if WORK_START <= t <= WORK_END:
            productive_minutes += DETECTION_INTERVAL_MINUTES
        else:
            unproductive_minutes += DETECTION_INTERVAL_MINUTES

    return {
        "tingkat_kehadiran": Indicator(
            label="Tingkat Kehadiran",
            value=tingkat_kehadiran,
            description="Hadir jika muncul >= 1x selama jam kerja"
        ),
        "ketepatan_datang": Indicator(
            label="Ketepatan Waktu Datang",
            value=ketepatan_datang,
            description="Waktu pertama muncul dalam jam kerja (HH:MM)"
        ),
        "lama_bekerja": Indicator(
            label="Lama Berada di Kantor",
            value=lama_bekerja,
            unit="minutes",
            description="Durasi dari first_seen sampai last_seen dalam jam kerja"
        ),
        "waktu_produktif": Indicator(
            label="Waktu Produktif",
            value=productive_minutes,
            unit="minutes",
            description="Menit occupied selama jam kerja"
        ),
        "waktu_tidak_produktif": Indicator(
            label="Waktu Tidak Produktif",
            value=unproductive_minutes,
            unit="minutes",
            description="Menit occupied di luar jam kerja"
        ),
    }


def compute_stats(person: str, target_date: date) -> StatsResponse:
    """Compute 5 management indicators for a person on a given date."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT image_timestamp, occupied
        FROM detection_logs
        WHERE person = %s
          AND DATE(image_timestamp) = %s
        ORDER BY image_timestamp ASC
    """
    cursor.execute(query, (person, target_date.isoformat()))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        return _default_stats(person, target_date)

    detections: list[tuple[datetime, bool]] = []
    for row in rows:
        dt = _parse_datetime(row["image_timestamp"])
        if dt:
            detections.append((dt, bool(row["occupied"])))

    if not detections:
        return _default_stats(person, target_date)

    work_start = _get_work_start()
    work_end = _get_work_end()
    interval = _get_interval_minutes()

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and work_start <= dt.time() <= work_end
    ]
    all_times = [dt for dt, _ in detections if work_start <= dt.time() <= work_end]

    work_hours = WorkHours(
        start=work_start.strftime("%H:%M"),
        end=work_end.strftime("%H:%M"),
        total_minutes=int((datetime.combine(target_date, work_end) - datetime.combine(target_date, work_start)).total_seconds() / 60)
    )
    return StatsResponse(
        person=person,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators=_build_indicators(occupied_in_work_hours, detections, all_times),
    )


def compute_desk_stats(desk: str, target_date: date) -> StatsResponse:
    """Compute 5 management indicators for a desk on a given date."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT image_timestamp, occupied
        FROM detection_logs
        WHERE desk = %s
          AND DATE(image_timestamp) = %s
        ORDER BY image_timestamp ASC
    """
    cursor.execute(query, (desk, target_date.isoformat()))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        return _default_stats(desk, target_date)

    detections: list[tuple[datetime, bool]] = []
    for row in rows:
        dt = _parse_datetime(row["image_timestamp"])
        if dt:
            detections.append((dt, bool(row["occupied"])))

    if not detections:
        return _default_stats(desk, target_date)

    work_start = _get_work_start()
    work_end = _get_work_end()

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and work_start <= dt.time() <= work_end
    ]
    all_times = [dt for dt, _ in detections if work_start <= dt.time() <= work_end]

    work_hours = WorkHours(
        start=work_start.strftime("%H:%M"),
        end=work_end.strftime("%H:%M"),
        total_minutes=600
    )
    return DeskStatsResponse(
        desk=desk,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators=_build_indicators(occupied_in_work_hours, detections, all_times),
    )


# ---------------------------------------------------------------------------
# Enhanced Stats Functions
# ---------------------------------------------------------------------------


def get_available_dates(person: str | None = None, year_month: str | None = None) -> list[str]:
    """
    Return sorted list of YYYY-MM-DD dates with detection data.
    If person is specified, filter by that person.
    If year_month is specified (YYYY-MM), filter to that month.
    Uses Asia/Jakarta timezone for date extraction.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if year_month:
        year, month = year_month.split("-")
        if person:
            query = """
                SELECT DISTINCT DATE(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) as jakarta_date
                FROM detection_logs
                WHERE person = %s
                  AND YEAR(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) = %s
                  AND MONTH(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) = %s
                ORDER BY jakarta_date DESC
            """
            cursor.execute(query, (person, int(year), int(month)))
        else:
            query = """
                SELECT DISTINCT DATE(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) as jakarta_date
                FROM detection_logs
                WHERE YEAR(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) = %s
                  AND MONTH(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) = %s
                ORDER BY jakarta_date DESC
            """
            cursor.execute(query, (int(year), int(month)))
    else:
        if person:
            query = """
                SELECT DISTINCT DATE(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) as jakarta_date
                FROM detection_logs
                WHERE person = %s
                ORDER BY jakarta_date DESC
            """
            cursor.execute(query, (person,))
        else:
            query = """
                SELECT DISTINCT DATE(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) as jakarta_date
                FROM detection_logs
                ORDER BY jakarta_date DESC
            """
            cursor.execute(query)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [row[0].isoformat() for row in rows]


def compute_daily_stats(person: str, target_date: date) -> DailyStats:
    """
    Per-person-per-day aggregation.
    - has_data: rows exist for this person+date
    - presence_status: NO_DATA / ABSENT / PRESENT
    - first_seen: first occupied timestamp during work hours
    - last_seen: last occupied timestamp during work hours
    - work_minutes: min((last_seen - first_seen), work_window_minutes)
    - productive_minutes: occupied_count_in_work_hours × interval
    - idle_minutes: occupied_count_outside_work_hours × interval
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT image_timestamp, occupied, confidence
        FROM detection_logs
        WHERE person = %s
          AND DATE(DATE_ADD(image_timestamp, INTERVAL 7 HOUR)) = %s
        ORDER BY image_timestamp ASC
    """
    cursor.execute(query, (person, target_date.isoformat()))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        return DailyStats(
            person=person,
            date=target_date.isoformat(),
            has_data=False,
            presence_status=PresenceStatus.NO_DATA,
            first_seen=None,
            last_seen=None,
            work_minutes=0,
            productive_minutes=0,
            idle_minutes=0,
        )

    detections: list[tuple[datetime, bool]] = []
    for row in rows:
        dt = _parse_datetime(row["image_timestamp"])
        if dt:
            detections.append((dt, bool(row["occupied"]) and row["confidence"] >= _get_presence_threshold()))

    if not detections:
        return DailyStats(
            person=person,
            date=target_date.isoformat(),
            has_data=True,
            presence_status=PresenceStatus.ABSENT,
            first_seen=None,
            last_seen=None,
            work_minutes=0,
            productive_minutes=0,
            idle_minutes=0,
        )

    work_start = _get_work_start()
    work_end = _get_work_end()
    interval = _get_interval_minutes()

    # Work window in minutes
    work_window_minutes = int((datetime.combine(target_date, work_end) - datetime.combine(target_date, work_start)).total_seconds() / 60)

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and work_start <= dt.time() <= work_end
    ]
    occupied_outside_work_hours = [
        dt for dt, occ in detections
        if occ and not (work_start <= dt.time() <= work_end)
    ]
    all_times_in_work = [dt for dt, _ in detections if work_start <= dt.time() <= work_end]

    if occupied_in_work_hours:
        presence_status = PresenceStatus.PRESENT
        first_seen = min(occupied_in_work_hours).strftime("%H:%M")
        last_seen = max(occupied_in_work_hours).strftime("%H:%M")
        first_dt = min(occupied_in_work_hours)
        last_dt = max(occupied_in_work_hours)
        diff_minutes = int((last_dt - first_dt).total_seconds() / 60)
        work_minutes = min(diff_minutes, work_window_minutes)
    else:
        presence_status = PresenceStatus.ABSENT
        first_seen = None
        last_seen = None
        work_minutes = 0

    productive_minutes = len(occupied_in_work_hours) * interval
    idle_minutes = len(occupied_outside_work_hours) * interval

    return DailyStats(
        person=person,
        date=target_date.isoformat(),
        has_data=True,
        presence_status=presence_status,
        first_seen=first_seen,
        last_seen=last_seen,
        work_minutes=work_minutes,
        productive_minutes=productive_minutes,
        idle_minutes=idle_minutes,
    )


def compute_aggregate(
    person: str | None,
    start_date: date,
    end_date: date,
    scope: str,
) -> AggregateStats:
    """
    Roll up daily stats over a date range.
    - Averages computed ONLY over days where has_data=True
    - Attendance rate = days_present / total_days (days with data)
    - On-time: first_seen <= work_start + tolerance
    """
    # Get all dates in range
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)

    # Get person_map for default person list
    from config import get_config
    cfg = get_config()
    persons = [person] if person else list(cfg.person_map.values())

    # Deduplicate persons
    persons = list(dict.fromkeys(persons))

    daily_stats_list: list[DailyStats] = []

    if person:
        # Single person
        for d in all_dates:
            ds = compute_daily_stats(person, d)
            daily_stats_list.append(ds)
    else:
        # All persons combined — compute per person, aggregate
        for p in persons:
            for d in all_dates:
                ds = compute_daily_stats(p, d)
                daily_stats_list.append(ds)

    # Filter days with data
    days_with_data = [ds for ds in daily_stats_list if ds.has_data]
    days_present = [ds for ds in days_with_data if ds.presence_status == PresenceStatus.PRESENT]

    total_days = len(days_with_data)
    days_present_count = len(days_present)
    attendance_rate = days_present_count / total_days if total_days > 0 else 0.0

    # Averages over days with data
    avg_work_minutes = sum(ds.work_minutes for ds in days_with_data) / total_days if total_days > 0 else 0.0
    avg_productive_minutes = sum(ds.productive_minutes for ds in days_with_data) / total_days if total_days > 0 else 0.0
    avg_idle_minutes = sum(ds.idle_minutes for ds in days_with_data) / total_days if total_days > 0 else 0.0

    # On-time stats
    tolerance = _get_tolerance_minutes()
    work_start = _get_work_start()
    tolerance_time = time(work_start.hour, work_start.minute + tolerance)

    on_time_count = 0
    arrival_times: list[str] = []
    for ds in days_present:
        if ds.first_seen:
            h, m = map(int, ds.first_seen.split(":"))
            arrival_t = time(h, m)
            if arrival_t <= tolerance_time:
                on_time_count += 1
            arrival_times.append(ds.first_seen)

    on_time_rate = on_time_count / days_present_count if days_present_count > 0 else 0.0

    # Average arrival time
    avg_arrival_time = "N/A"
    if arrival_times:
        total_minutes = 0
        for at in arrival_times:
            h, m = map(int, at.split(":"))
            total_minutes += h * 60 + m
        avg_mins = total_minutes / len(arrival_times)
        avg_arrival_time = f"{int(avg_mins // 60):02d}:{int(avg_mins % 60):02d}"

    return AggregateStats(
        scope=scope,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        person=person,
        total_days=total_days,
        days_present=days_present_count,
        attendance_rate=round(attendance_rate, 3),
        avg_work_minutes=round(avg_work_minutes, 1),
        avg_productive_minutes=round(avg_productive_minutes, 1),
        avg_idle_minutes=round(avg_idle_minutes, 1),
        on_time_count=on_time_count,
        on_time_rate=round(on_time_rate, 3),
        avg_arrival_time=avg_arrival_time,
    )


def compute_trend_series(
    person: str | None,
    start_date: date,
    end_date: date,
) -> list[TrendPoint]:
    """
    Daily breakdown for trend charts.
    One TrendPoint per day in range (including days with no data, marked NO_DATA).
    """
    from config import get_config
    cfg = get_config()

    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)

    if person:
        persons = [person]
    else:
        persons = list(dict.fromkeys(cfg.person_map.values()))

    trend_points: list[TrendPoint] = []
    for p in persons:
        for d in all_dates:
            ds = compute_daily_stats(p, d)
            trend_points.append(TrendPoint(
                date=d.isoformat(),
                person=p,
                presence_status=ds.presence_status,
                work_minutes=ds.work_minutes,
                productive_minutes=ds.productive_minutes,
                arrival_time=ds.first_seen,
            ))

    return trend_points


def compute_person_comparison(
    start_date: date,
    end_date: date,
) -> list[PersonComparison]:
    """
    Per-person summary for 'Semua orang' view.
    One row per person in person_map.
    """
    from config import get_config
    cfg = get_config()
    persons = list(dict.fromkeys(cfg.person_map.values()))

    comparisons: list[PersonComparison] = []
    for p in persons:
        agg = compute_aggregate(person=p, start_date=start_date, end_date=end_date, scope="comparison")
        comparisons.append(PersonComparison(
            person=p,
            attendance_rate=agg.attendance_rate,
            avg_work_minutes=agg.avg_work_minutes,
            avg_arrival_time=agg.avg_arrival_time,
            on_time_rate=agg.on_time_rate,
        ))

    return comparisons
