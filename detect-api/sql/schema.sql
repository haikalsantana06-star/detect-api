-- Run this once on your shared hosting MySQL (via phpMyAdmin or CLI)

CREATE DATABASE IF NOT EXISTS upitas_mon;
USE upitas_mon;

-- Raw detection log — one row per ESP32-CAM capture
CREATE TABLE IF NOT EXISTS detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_timestamp DATETIME NOT NULL,
    person VARCHAR(50) NOT NULL,
    desk VARCHAR(20) NOT NULL,
    occupied TINYINT(1) NOT NULL COMMENT '0=empty, 1=occupied',
    confidence FLOAT NOT NULL,
    angle VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_person_date (person, image_timestamp),
    INDEX idx_desk_date (desk, image_timestamp),
    INDEX idx_timestamp (image_timestamp)
);

-- Aggregated daily summary per person
CREATE TABLE IF NOT EXISTS presence_summary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    first_seen TIME DEFAULT NULL,
    last_seen TIME DEFAULT NULL,
    total_presence_minutes INT DEFAULT 0,
    work_hours_minutes INT DEFAULT 600,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_person_date (person, date)
);
