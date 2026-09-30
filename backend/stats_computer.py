"""Compute 5 management indicators from detection_logs in MySQL."""
import logging
from datetime import date, datetime, time

from config import get_db_connection
from stats_schemas import DeskStatsResponse, Indicator, StatsResponse, WorkHours

logger = logging.getLogger(__name__)

WORK_START = time(7, 0)
WORK_END = time(17, 0)
DETECTION_INTERVAL_MINUTES = 5


def _parse_datetime(dt_val) -> datetime | None:
    """Convert MySQL datetime result to Python datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, str):
        return datetime.fromisoformat(dt_val)
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

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and WORK_START <= dt.time() <= WORK_END
    ]
    all_times = [dt for dt, _ in detections if WORK_START <= dt.time() <= WORK_END]

    work_hours = WorkHours(start="07:00", end="17:00", total_minutes=600)
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

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and WORK_START <= dt.time() <= WORK_END
    ]
    all_times = [dt for dt, _ in detections if WORK_START <= dt.time() <= WORK_END]

    work_hours = WorkHours(start="07:00", end="17:00", total_minutes=600)
    return DeskStatsResponse(
        desk=desk,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators=_build_indicators(occupied_in_work_hours, detections, all_times),
    )
