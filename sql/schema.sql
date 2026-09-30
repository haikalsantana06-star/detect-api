-- ============================================
-- CAPE - detection_logs table
-- Run this on your MySQL/MariaDB server
-- ============================================

CREATE TABLE IF NOT EXISTS detection_logs (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    image_timestamp DATETIME NOT NULL,
    person      VARCHAR(100) NOT NULL,
    desk        VARCHAR(50) NOT NULL,
    occupied    TINYINT(1) NOT NULL DEFAULT 0,
    confidence  FLOAT NOT NULL DEFAULT 0.0,
    angle       VARCHAR(50) NOT NULL DEFAULT 'angle_1',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for stats queries
    INDEX idx_person_timestamp (person, image_timestamp),
    INDEX idx_desk_timestamp   (desk, image_timestamp),
    INDEX idx_angle_timestamp    (angle, image_timestamp),
    INDEX idx_timestamp          (image_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- NOTES:
-- - `person` matches person_map in zones.json (e.g., "Asep", "Budi")
-- - `desk` matches zone names (e.g., "desk_a", "desk_b")
-- - `angle` matches camera angle (e.g., "angle_1", "angle_2")
-- - `occupied` = 1 means someone is sitting, 0 means empty
-- - `confidence` is the model probability (0.0 - 1.0)
-- ============================================

-- To drop and recreate (CAREFUL - destroys all data):
-- DROP TABLE IF EXISTS detection_logs;
