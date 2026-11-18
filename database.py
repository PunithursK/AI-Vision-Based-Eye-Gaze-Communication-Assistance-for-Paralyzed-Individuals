"""
Database management module for AI-Vision-Based Communication Assistance.
Stores caregiver configurations, system parameters, and full request history logs.
"""

import sqlite3
import os
import json
import csv
import io
import re
import secrets
import time
from datetime import datetime

DB_FILE = os.environ.get("GAZE_DB_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "paralyzed_assistance.db"))

DEFAULT_CONFIG = {
    "caregiver_name": "Primary Caregiver",
    "caregiver_email": "caregiver.alert@example.com",
    "caregiver_phone": "+1-555-0199",
    "patient_name": "Patient Alpha (Room 304)",
    "email_provider": "brevo",
    "brevo_api_key": "",
    "brevo_sender_name": "Eye Gaze AI",
    "brevo_sender_email": "ogjoyboy21@gmail.com",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 465,
    "smtp_user": "",
    "smtp_pass": "",
    "use_tls": 1,
    "dwell_time_seconds": 3.0,
    "calibration_dwell_seconds": 1.5,
    "head_sensitivity": 1.2,
    "eye_sensitivity": 1.5,
    "sound_enabled": 1,
    "voice_enabled": 0,  # Voice assistance disabled per user request
    "simulation_mode": 1,  # When 1, simulates SMTP if credentials are blank without throwing errors
    "tracking_mode": "both_eyes",  # Both-Eye simultaneous tracking mandatory architecture
    "camera_index": 0,  # System Camera (0) until external webcam is detected/selected
    "mirror_view": 1  # Mirrored camera view for natural user perspective
}

ALLOWED_CONFIG_KEYS = {
    "caregiver_name", "caregiver_email", "caregiver_phone", "patient_name",
    "email_provider", "brevo_api_key", "brevo_sender_name", "brevo_sender_email",
    "smtp_host", "smtp_port", "smtp_user", "smtp_pass", "use_tls",
    "dwell_time_seconds", "calibration_dwell_seconds", "head_sensitivity", "eye_sensitivity",
    "sound_enabled", "voice_enabled", "simulation_mode", "tracking_mode",
    "camera_index", "mirror_view"
}


def get_db_connection():
    """Create a database connection with dictionary-like row access, timeout, and WAL mode."""
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def init_db():
    """Initialize database tables and default configuration (idempotent, guarded by schema migration version)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Safe, idempotent migration: drop obsolete caregivers account table if present
    cursor.execute("DROP TABLE IF EXISTS caregivers")

    # Table for caregiver & system configuration
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS caregiver_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        caregiver_name TEXT NOT NULL,
        caregiver_email TEXT NOT NULL,
        caregiver_phone TEXT,
        patient_name TEXT,
        email_provider TEXT DEFAULT 'brevo',
        brevo_api_key TEXT DEFAULT '',
        brevo_sender_name TEXT DEFAULT 'Eye Gaze AI',
        brevo_sender_email TEXT DEFAULT 'ogjoyboy21@gmail.com',
        smtp_host TEXT,
        smtp_port INTEGER,
        smtp_user TEXT,
        smtp_pass TEXT,
        use_tls INTEGER DEFAULT 1,
        dwell_time_seconds REAL DEFAULT 3.0,
        calibration_dwell_seconds REAL DEFAULT 1.5,
        head_sensitivity REAL DEFAULT 1.2,
        eye_sensitivity REAL DEFAULT 1.5,
        sound_enabled INTEGER DEFAULT 1,
        voice_enabled INTEGER DEFAULT 0,
        simulation_mode INTEGER DEFAULT 1,
        tracking_mode TEXT DEFAULT 'both_eyes',
        camera_index INTEGER DEFAULT 0,
        mirror_view INTEGER DEFAULT 1,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Check for column migrations
    cursor.execute("PRAGMA table_info(caregiver_config)")
    columns = [row["name"] for row in cursor.fetchall()]
    for col, ctype, cdef in [
        ("tracking_mode", "TEXT", "'both_eyes'"),
        ("camera_index", "INTEGER", "0"),
        ("mirror_view", "INTEGER", "1"),
        ("calibration_dwell_seconds", "REAL", "1.5"),
        ("email_provider", "TEXT", "'brevo'"),
        ("brevo_api_key", "TEXT", "''"),
        ("brevo_sender_name", "TEXT", "'Eye Gaze AI'"),
        ("brevo_sender_email", "TEXT", "'ogjoyboy21@gmail.com'")
    ]:
        if col not in columns:
            try:
                cursor.execute(f"ALTER TABLE caregiver_config ADD COLUMN {col} {ctype} DEFAULT {cdef}")
            except Exception:
                pass

    # Table for historical request logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS requests_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        block_id TEXT NOT NULL,
        block_title TEXT NOT NULL,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        message TEXT NOT NULL,
        caregiver_email TEXT NOT NULL,
        status TEXT NOT NULL, -- 'PENDING', 'SENDING', 'DELIVERED', 'SIMULATED', 'FAILED'
        delivery_duration_ms INTEGER DEFAULT 0,
        notes TEXT
    );
    """)

    # Table for calibration results and history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calibration_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL,
        model_name TEXT,
        num_points INTEGER DEFAULT 5,
        valid_samples INTEGER DEFAULT 0,
        calibration_error_px REAL,
        mean_gaze_error_px REAL,
        val_rmse_px REAL,
        validation_accuracy REAL,
        is_calibrated INTEGER DEFAULT 0,
        quality TEXT,
        details TEXT
    );
    """)

    # Insert default config if not present
    cursor.execute("SELECT COUNT(*) FROM caregiver_config WHERE id = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO caregiver_config (
            id, caregiver_name, caregiver_email, caregiver_phone, patient_name,
            email_provider, brevo_api_key, brevo_sender_name, brevo_sender_email,
            smtp_host, smtp_port, smtp_user, smtp_pass, use_tls,
            dwell_time_seconds, calibration_dwell_seconds, head_sensitivity, eye_sensitivity,
            sound_enabled, voice_enabled, simulation_mode, tracking_mode, camera_index, mirror_view
        ) VALUES (1, :caregiver_name, :caregiver_email, :caregiver_phone, :patient_name,
                  :email_provider, :brevo_api_key, :brevo_sender_name, :brevo_sender_email,
                  :smtp_host, :smtp_port, :smtp_user, :smtp_pass, :use_tls,
                  :dwell_time_seconds, :calibration_dwell_seconds, :head_sensitivity, :eye_sensitivity,
                  :sound_enabled, :voice_enabled, :simulation_mode, :tracking_mode, :camera_index, :mirror_view)
        """, DEFAULT_CONFIG)

    # One-time legacy cleanup guarded by PRAGMA user_version to prevent overriding user settings repeatedly (F2)
    cursor.execute("PRAGMA user_version")
    user_ver = cursor.fetchone()[0]
    if user_ver < 2:
        cursor.execute("PRAGMA user_version = 2")

    conn.commit()
    conn.close()


def get_config(include_secrets: bool = False):
    """
    Retrieve current configuration.
    F3: Excludes raw credentials by default so API responses and templates never expose plaintext secrets.
    Checks environment variables (BREVO_API_KEY, BREVO_SENDER_NAME, BREVO_SENDER_EMAIL) with precedence over DB.
    """
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM caregiver_config WHERE id = 1").fetchone()
    conn.close()
    if row:
        config_dict = dict(row)
    else:
        config_dict = DEFAULT_CONFIG.copy()

    # Environment variable overrides take precedence
    env_api_key = os.environ.get("BREVO_API_KEY")
    if env_api_key and env_api_key.strip():
        config_dict["brevo_api_key"] = env_api_key.strip()

    env_sender_name = os.environ.get("BREVO_SENDER_NAME")
    if env_sender_name and env_sender_name.strip():
        config_dict["brevo_sender_name"] = env_sender_name.strip()

    env_sender_email = os.environ.get("BREVO_SENDER_EMAIL")
    if env_sender_email and env_sender_email.strip():
        config_dict["brevo_sender_email"] = env_sender_email.strip()

    raw_pass = config_dict.get("smtp_pass") or ""
    config_dict["has_smtp_pass"] = bool(raw_pass)
    config_dict["smtp_pass_masked"] = "••••••••••••" if raw_pass else ""

    raw_api_key = config_dict.get("brevo_api_key") or ""
    config_dict["has_brevo_api_key"] = bool(raw_api_key)
    config_dict["brevo_api_key_masked"] = "••••••••••••" if raw_api_key else ""

    if not include_secrets:
        config_dict.pop("smtp_pass", None)
        config_dict.pop("brevo_api_key", None)
    return config_dict


def get_email_credentials():
    """
    Retrieve full email credentials securely for internal email dispatching only.
    Checks environment variables (BREVO_API_KEY, BREVO_SENDER_NAME, BREVO_SENDER_EMAIL, GAZE_CAREGIVER_EMAIL)
    with precedence over database configuration.
    """
    cfg = get_config(include_secrets=True)
    env_api_key = os.environ.get("BREVO_API_KEY")
    if env_api_key and env_api_key.strip():
        cfg["brevo_api_key"] = env_api_key.strip()
    env_sender_name = os.environ.get("BREVO_SENDER_NAME")
    if env_sender_name and env_sender_name.strip():
        cfg["brevo_sender_name"] = env_sender_name.strip()
    env_sender_email = os.environ.get("BREVO_SENDER_EMAIL")
    if env_sender_email and env_sender_email.strip():
        cfg["brevo_sender_email"] = env_sender_email.strip()
    env_caregiver = os.environ.get("GAZE_CAREGIVER_EMAIL")
    if env_caregiver and env_caregiver.strip():
        cfg["caregiver_email"] = env_caregiver.strip()
    # Backward compatibility for legacy SMTP env vars
    env_sender = os.environ.get("GAZE_EMAIL_SENDER")
    if env_sender and env_sender.strip():
        cfg["smtp_user"] = env_sender.strip()
    env_pass = os.environ.get("GAZE_EMAIL_PASSWORD")
    if env_pass and env_pass.strip():
        cfg["smtp_pass"] = env_pass.strip()
    return cfg


get_smtp_credentials = get_email_credentials


def update_config(data: dict):
    """
    F1: Partial update of configuration — only keys present in data are validated and updated.
    Preserves all unspecified user settings.
    """
    if not isinstance(data, dict):
        return get_config()

    conn = get_db_connection()
    current_row = conn.execute("SELECT * FROM caregiver_config WHERE id = 1").fetchone()
    if not current_row:
        conn.close()
        init_db()
        conn = get_db_connection()
        current_row = conn.execute("SELECT * FROM caregiver_config WHERE id = 1").fetchone()

    current = dict(current_row) if current_row else DEFAULT_CONFIG.copy()

    updates = {}
    for key, value in data.items():
        if key not in ALLOWED_CONFIG_KEYS:
            continue

        if key == "brevo_api_key":
            # If masked with bullets or empty string, preserve existing saved key
            if value and "•" not in str(value):
                updates["brevo_api_key"] = str(value).strip()
        elif key in ("email_provider", "brevo_sender_name", "brevo_sender_email"):
            updates[key] = str(value).strip()
        elif key == "smtp_pass":
            # If masked with bullets or empty string, preserve existing password
            if value and "•" not in str(value):
                updates["smtp_pass"] = str(value).replace(" ", "").strip()
        elif key == "smtp_port":
            try:
                port_val = int(value)
                updates["smtp_port"] = max(1, min(65535, port_val))
            except (ValueError, TypeError):
                pass
        elif key == "dwell_time_seconds":
            try:
                dwell_val = float(value)
                updates["dwell_time_seconds"] = max(0.5, min(10.0, dwell_val))
            except (ValueError, TypeError):
                pass
        elif key == "calibration_dwell_seconds":
            try:
                calib_val = float(value)
                updates["calibration_dwell_seconds"] = max(0.8, min(5.0, calib_val))
            except (ValueError, TypeError):
                pass
        elif key in ("head_sensitivity", "eye_sensitivity"):
            try:
                sens_val = float(value)
                updates[key] = max(0.1, min(5.0, sens_val))
            except (ValueError, TypeError):
                pass
        elif key in ("sound_enabled", "voice_enabled", "simulation_mode", "use_tls", "mirror_view"):
            if isinstance(value, str):
                updates[key] = 1 if value.strip().lower() in ("1", "true", "yes", "on") else 0
            else:
                updates[key] = 1 if bool(value) else 0
        elif key == "camera_index":
            try:
                updates["camera_index"] = max(0, int(value))
            except (ValueError, TypeError):
                pass
        elif key in ("caregiver_name", "caregiver_email", "caregiver_phone", "patient_name", "smtp_host", "smtp_user", "tracking_mode"):
            updates[key] = str(value).strip()

    if updates:
        set_clauses = [f"{col} = ?" for col in updates.keys()]
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        values = list(updates.values())
        sql = f"UPDATE caregiver_config SET {', '.join(set_clauses)} WHERE id = 1"
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()

    conn.close()
    return get_config()


def log_request(block_id, block_title, category, priority, message, caregiver_email, status="PENDING", notes=""):
    """Insert a new patient communication request into the log table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO requests_log (
        block_id, block_title, category, priority, message, caregiver_email, status, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (block_id, block_title, category, priority, message, caregiver_email, status, notes))
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return request_id


def update_request_status(request_id, status, duration_ms=0, notes=None):
    """Update status and delivery duration of an existing request."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if notes is not None:
        cursor.execute("""
        UPDATE requests_log SET status = ?, delivery_duration_ms = ?, notes = ? WHERE id = ?
        """, (status, duration_ms, notes, request_id))
    else:
        cursor.execute("""
        UPDATE requests_log SET status = ?, delivery_duration_ms = ? WHERE id = ?
        """, (status, duration_ms, request_id))
    conn.commit()
    conn.close()


def get_requests_log(limit=100):
    """Retrieve recent request logs sorted newest first."""
    conn = get_db_connection()
    try:
        limit_val = max(1, min(1000, int(limit)))
    except (ValueError, TypeError):
        limit_val = 100
    rows = conn.execute("""
    SELECT * FROM requests_log ORDER BY id DESC LIMIT ?
    """, (limit_val,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_request_by_id(request_id):
    """Retrieve a single patient request log by primary key ID."""
    try:
        req_id = int(request_id)
    except (ValueError, TypeError):
        return None
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM requests_log WHERE id = ?", (req_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def clear_requests_log():
    """Clear all records from the request log table."""
    conn = get_db_connection()
    conn.execute("DELETE FROM requests_log")
    conn.commit()
    conn.close()
    return True


def reset_request_id_sequence():
    """
    Safely reset the auto-increment counter sequence for requests_log table within a transaction.
    Preserves all existing patient request records.
    - Resets sqlite_sequence entry for 'requests_log'.
    - If the table has no records, the next inserted record will receive ID 1.
    - If the table contains existing records, SQLite prevents duplicate IDs (ensuring no overwrites
      and no duplicate key collisions), and an advisory message explains that restarting strictly from 1
      requires clearing existing history records first.
    """
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM requests_log")
            row = cursor.fetchone()
            count = row[0] if row else 0
            max_id = row[1] if row else 0

            cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'requests_log'")

        if count == 0:
            return {
                "success": True,
                "has_records": False,
                "record_count": 0,
                "next_id": 1,
                "message": "Request ID counter reset successfully. The next patient request will start from ID #1."
            }
        else:
            return {
                "success": True,
                "has_records": True,
                "record_count": count,
                "next_id": max_id + 1,
                "message": (
                    f"Request ID counter reset. All {count} existing records preserved. "
                    f"To prevent duplicate IDs, the next record will use ID #{max_id + 1}. "
                    "To restart strictly from ID #1, please clear existing history records first."
                )
            }
    finally:
        conn.close()


def export_logs_csv():
    """Export all request logs as a formatted CSV string."""
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM requests_log ORDER BY id DESC").fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp (UTC)", "Block ID", "Request Title", "Category",
        "Priority", "Message", "Caregiver Email", "Delivery Status",
        "Duration (ms)", "Notes"
    ])
    for r in rows:
        writer.writerow([
            r["id"], r["timestamp"], r["block_id"], r["block_title"],
            r["category"], r["priority"], r["message"], r["caregiver_email"],
            r["status"], r["delivery_duration_ms"], r["notes"]
        ])
    return output.getvalue()


def save_calibration_result(data: dict):
    """
    Save calibration accuracy and error results to database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO calibration_history (
        status, model_name, num_points, valid_samples,
        calibration_error_px, mean_gaze_error_px, val_rmse_px,
        validation_accuracy, is_calibrated, quality, details
    ) VALUES (
        :status, :model_name, :num_points, :valid_samples,
        :calibration_error_px, :mean_gaze_error_px, :val_rmse_px,
        :validation_accuracy, :is_calibrated, :quality, :details
    )
    """, {
        "status": data.get("status", "Completed"),
        "model_name": data.get("model_name", "None"),
        "num_points": int(data.get("num_points", 5)),
        "valid_samples": int(data.get("valid_samples", 0)),
        "calibration_error_px": data.get("calibration_error_px"),
        "mean_gaze_error_px": data.get("mean_gaze_error_px"),
        "val_rmse_px": data.get("val_rmse_px"),
        "validation_accuracy": data.get("validation_accuracy"),
        "is_calibrated": 1 if data.get("is_calibrated") else 0,
        "quality": data.get("quality", "None"),
        "details": data.get("details", "")
    })
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def get_latest_calibration_result():
    """
    Retrieve the most recent calibration result from the database.
    If none exists in the DB but a valid calibration_profile.json exists on disk,
    migrates it into the DB for seamless continuity.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, timestamp, status, model_name, num_points, valid_samples,
           calibration_error_px, mean_gaze_error_px, val_rmse_px,
           validation_accuracy, is_calibrated, quality, details
    FROM calibration_history
    ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)

    # Check for existing profile on disk
    profile_path = os.environ.get("CALIBRATION_PROFILE_PATH", "calibration_profile.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                prof = json.load(f)
            if prof.get("source") == "user_calibration":
                rec = {
                    "status": "Completed" if prof.get("has_trained_model") else "Calibrated",
                    "model_name": prof.get("model_name", "Affine Standard"),
                    "num_points": 5,
                    "valid_samples": prof.get("samples_count", 0),
                    "calibration_error_px": prof.get("val_mae_px"),
                    "mean_gaze_error_px": prof.get("val_mae_px"),
                    "val_rmse_px": prof.get("val_rmse_px"),
                    "validation_accuracy": prof.get("accuracy") or prof.get("hit_rate"),
                    "is_calibrated": 1 if prof.get("has_trained_model") else 0,
                    "quality": prof.get("quality", "Restored Profile"),
                    "details": json.dumps(prof)
                }
                save_calibration_result(rec)
                return get_latest_calibration_result()
        except Exception:
            pass

    return None


# Initialize upon module import
init_db()
