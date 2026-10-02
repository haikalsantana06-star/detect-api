-- CAPE detection_logs table
-- Stores each desk occupancy detection result

CREATE TABLE IF NOT EXISTS detection_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_timestamp DATETIME NOT NULL,
    person VARCHAR(100) NOT NULL,
    desk VARCHAR(50) NOT NULL,
    occupied TINYINT(1) NOT NULL DEFAULT 0,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    angle VARCHAR(50) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_person_timestamp (person, image_timestamp),
    INDEX idx_desk_timestamp (desk, image_timestamp),
    INDEX idx_angle (angle)
);
