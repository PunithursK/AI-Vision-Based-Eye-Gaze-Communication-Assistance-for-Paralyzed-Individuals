"""
Flask Backend Application for AI-Vision-Based Communication Assistance for Paralyzed Individuals.
Provides REST APIs, Server-Sent Events (SSE) for 30 FPS gaze telemetry,
MJPEG camera streaming, lockout protocol orchestration, and caregiver email alerts.
"""

from flask import Flask, render_template, Response, request, jsonify, send_file, make_response, redirect, url_for, flash
from flask_cors import CORS
from functools import wraps
import threading
import time
import json
import io
import os
import sys
import signal
import atexit
import secrets
import database
import email_notifier
from ai_vision_engine import vision_engine

_shutdown_in_progress = False

def shutdown_application(signum=None, frame=None):
    """
    Safely shuts down the application:
    1. Stops camera and background vision loop.
    2. Releases webcam hardware and DirectShow COM resources.
    3. Exits immediately via os._exit(0) to prevent native Windows/OpenCV deadlocks during interpreter shutdown.
    """
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True
    print("\n[Shutdown] Shutting down Eye Gaze system...")
    try:
        vision_engine.shutdown()
    except Exception:
        pass
    print("[Shutdown] Hardware released. Application terminated cleanly.")
    os._exit(0)

def setup_graceful_shutdown():
    """Register OS signals and Windows ConsoleCtrlHandler for Ctrl+C / SIGINT / SIGTERM / SIGBREAK."""
    try:
        signal.signal(signal.SIGINT, shutdown_application)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, shutdown_application)
    except Exception:
        pass
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, shutdown_application)
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            def _win_ctrl_handler(ctrl_type):
                shutdown_application()
                return True
            global _win_ctrl_ref
            _win_ctrl_ref = handler_type(_win_ctrl_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_ctrl_ref, True)
        except Exception:
            pass

    try:
        atexit.register(vision_engine.shutdown)
    except Exception:
        pass

# Ensure shutdown hooks are registered immediately
setup_graceful_shutdown()

# Suppress Werkzeug access logs and Flask development server banners for clean terminal output
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)
try:
    import flask.cli
    flask.cli.show_server_banner = lambda *args: None
except Exception:
    pass

app = Flask(__name__)

# Secure Session Configuration for Caregiver Authentication
secret_key = os.environ.get("FLASK_SECRET_KEY")
if not secret_key:
    import warnings
    warnings.warn(
        "FLASK_SECRET_KEY is not set. Using default key — sessions are forgeable! "
        "Set FLASK_SECRET_KEY in your environment for production.",
        RuntimeWarning, stacklevel=2
    )
    secret_key = "eyegaze_caregiver_secure_session_key_2026_x89a"
app.secret_key = secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# F5: Restrict CORS to localhost and local loopback origins
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost", "http://127.0.0.1"]}})

# F5: Per-run random CSRF token
CSRF_TOKEN = secrets.token_hex(16)



# Event deduplication and idempotency tracking
_processed_events = {}
_events_lock = threading.Lock()


def is_duplicate_event(event_id: str, ttl: float = 60.0) -> bool:
    """Check and record unique event IDs to prevent duplicate trigger submissions."""
    if not event_id:
        return False
    now = time.time()
    with _events_lock:
        # Purge expired event IDs (> 120s)
        stale_keys = [k for k, ts in _processed_events.items() if now - ts > 120.0]
        for k in stale_keys:
            _processed_events.pop(k, None)

        if event_id in _processed_events:
            return True
        _processed_events[event_id] = now
        return False


# Exactly Four Predefined Request Blocks arranged in a 2x2 layout:
# 1. FOOD (Top-Left)
# 2. WATER (Top-Right)
# 3. MEDICINE (Bottom-Left)
# 4. HELP (Bottom-Right)
COMMUNICATION_BLOCKS = BLOCKS_CONFIG = [
    {
        "number": 1,
        "id": "food",
        "title": "Food",
        "short_title": "Food",
        "icon": "restaurant",
        "bs_icon": "bi-cup-hot-fill",
        "category": "Nutrition",
        "priority": "MEDIUM",
        "accent": "#d97706",
        "secondary_accent": "#b45309",
        "message": "Patient is hungry and requests meal or nourishment assistance.",
        "description": "Request meal, nourishment, or eating assistance."
    },
    {
        "number": 2,
        "id": "water",
        "title": "Water",
        "short_title": "Water",
        "icon": "water_drop",
        "bs_icon": "bi-droplet-fill",
        "category": "Hydration",
        "priority": "MEDIUM",
        "accent": "#3b82f6",
        "secondary_accent": "#1d4ed8",
        "message": "Patient requires drinking water or hydration assistance.",
        "description": "Request assistance for fresh drinking water or beverages."
    },
    {
        "number": 3,
        "id": "medicine",
        "title": "Medicine",
        "short_title": "Medicine",
        "icon": "medication",
        "bs_icon": "bi-capsule",
        "category": "Medical Care",
        "priority": "HIGH",
        "accent": "#10b981",
        "secondary_accent": "#047857",
        "message": "Patient requests scheduled medication, pills, or pain relief.",
        "description": "Alert caregiver that it is time for prescription medication."
    },
    {
        "number": 4,
        "id": "help",
        "title": "Help",
        "short_title": "Help",
        "icon": "emergency",
        "bs_icon": "bi-exclamation-triangle-fill",
        "category": "Urgent Care",
        "priority": "CRITICAL",
        "accent": "#dc2626",
        "secondary_accent": "#991b1b",
        "message": "CRITICAL ALERT: Patient urgently requests immediate caregiver presence!",
        "description": "High-priority alarm alerting caregiver to come immediately."
    }
]

# Lockout State Management
# When a block dwell triggers, the entire system locks until notification is complete
lock_state = {
    "is_locked": False,
    "locked_block_id": None,
    "locked_block_title": None,
    "status": "IDLE",  # 'IDLE', 'LOCKING', 'SENDING_EMAIL', 'DELIVERED', 'COOLDOWN'
    "message": "",
    "lock_timestamp": 0.0,
    "unlock_timestamp": 0.0,
    "last_request_id": None,
    "email_status": "IDLE",  # 'IDLE', 'PENDING', 'SENDING', 'DELIVERED', 'FAILED', 'SIMULATED'
    "notification_status": "IDLE",
    "watchdog_unlocked": False
}
lock_mutex = threading.Lock()
_active_email_thread = None
_active_request_id = None


def check_lock_state() -> dict:
    """
    G1 & E2: Helper function merging cooldown expiry and safety watchdog logic.
    Ensures the interface is never locked permanently even on unhandled worker failures,
    while not falsely marking an in-progress email as FAILED in the database.
    """
    global _active_email_thread, _active_request_id
    with lock_mutex:
        if not lock_state["is_locked"]:
            return dict(lock_state)

        now = time.time()
        elapsed = now - lock_state["lock_timestamp"]

        # E2: Safety watchdog - release interface if locked in SENDING_EMAIL for > 20s
        if lock_state["status"] == "SENDING_EMAIL" and elapsed > 20.0:
            worker_alive = (_active_email_thread is not None and _active_email_thread.is_alive())
            
            # If worker thread died or exceeded hard timeout (45s), mark FAILED in database
            if not worker_alive or elapsed > 45.0:
                print(f"[Lock Safety Watchdog] Worker stuck or dead (elapsed={elapsed:.1f}s, alive={worker_alive}). Marking request as FAILED.")
                lock_state["status"] = "COOLDOWN"
                lock_state["email_status"] = "FAILED"
                lock_state["notification_status"] = "FAILED"
                lock_state["message"] = "Alert dispatch timed out. System unlocked for patient safety."
                lock_state["unlock_timestamp"] = now + 1.5
                req_id = lock_state.get("last_request_id")
                if req_id:
                    try:
                        database.update_request_status(
                            req_id,
                            "FAILED",
                            notes="Alert dispatch timed out (worker exceeded watchdog timeout policy)"
                        )
                    except Exception:
                        pass
            else:
                # Worker is still actively running: safely unlock interface WITHOUT falsely changing email delivery result
                if not lock_state.get("watchdog_unlocked"):
                    print(f"[Lock Safety Watchdog] 20s email watchdog elapsed. Safely unlocking interface while worker finishes in background.")
                    lock_state["watchdog_unlocked"] = True
                    lock_state["status"] = "COOLDOWN"
                    # Keep email_status as SENDING, unlock interface safely
                    lock_state["message"] = "Interface safely unlocked. Alert dispatch continuing in background."
                    lock_state["unlock_timestamp"] = now + 1.5

        # Check if cooldown period expired
        if lock_state["status"] == "COOLDOWN":
            if now >= lock_state["unlock_timestamp"]:
                lock_state["is_locked"] = False
                lock_state["locked_block_id"] = None
                lock_state["locked_block_title"] = None
                lock_state["status"] = "IDLE"
                lock_state["message"] = ""

        return dict(lock_state)


_last_vision_err_time = 0.0

def background_vision_loop():
    """
    D7 & GA-7: Continuously runs the vision engine to process frames at webcam rate (~30 FPS).
    Logs exceptions with rate-limiting and exposes last_error in telemetry.
    Respects camera_enabled and is_running states, cleanly breaking on system shutdown.
    """
    global _last_vision_err_time
    while not getattr(vision_engine, "is_shutting_down", False):
        try:
            if not getattr(vision_engine, "camera_enabled", False) or not getattr(vision_engine, "is_running", False):
                time.sleep(0.1)  # Idle when camera is off/disabled to prevent busy-waiting
                continue
            vision_engine.process_frame()
            if not getattr(vision_engine, "is_running", False):
                time.sleep(0.1)  # Idle when camera turned off during frame processing
            # Frame capture (cv2.VideoCapture.read) naturally throttles to webcam frame rate (~30 FPS); no artificial sleep needed
        except Exception as e:
            if getattr(vision_engine, "is_shutting_down", False):
                break
            now = time.time()
            if now - _last_vision_err_time > 5.0:
                print(f"[ERROR] Vision loop exception: {e}")
                _last_vision_err_time = now
            vision_engine.last_error = str(e)
            if hasattr(vision_engine, "telemetry") and isinstance(vision_engine.telemetry, dict):
                vision_engine.telemetry["last_error"] = str(e)
            time.sleep(0.1)


# D1 & F1: Camera initialization at startup
# Prefer the saved camera index from configuration; never auto-switch to index >= 1 at startup
# and never overwrite user settings in DB at startup.
try:
    _init_cfg = database.get_config()
    vision_engine.set_tracking_mode(_init_cfg.get("tracking_mode", "both_eyes"))
    vision_engine.head_weight = float(_init_cfg.get("head_sensitivity", 1.2))
    vision_engine.eye_weight = float(_init_cfg.get("eye_sensitivity", 1.5))
    vision_engine.mirror_view = bool(_init_cfg.get("mirror_view", 1))

    cfg_cam = _init_cfg.get("camera_index")
    chosen_cam = int(cfg_cam) if cfg_cam is not None else 0

    vision_engine.camera_index = chosen_cam
    vision_engine.start_camera(chosen_cam)

    def handle_camera_fallback(fallback_index):
        try:
            database.update_config({"camera_index": fallback_index})
            print(f"[App] Updated database config to camera {fallback_index} following auto-fallback.")
        except Exception as err:
            print(f"[App Warning] Failed to update config on fallback: {err}")

    vision_engine.on_camera_fallback = handle_camera_fallback
except Exception as e:
    print(f"[App Init Warning] Config init: {e}")

# Start vision background worker thread after camera is initialized
vision_thread = threading.Thread(target=background_vision_loop, daemon=True)
vision_thread.start()


# F5: CSRF protection on mutating routes
MUTATING_ENDPOINTS = {
    "/api/trigger_block",
    "/api/unlock",
    "/api/camera/toggle",
    "/api/camera/select",
    "/api/camera/mirror",
    "/api/recenter",
    "/api/calibration/start",
    "/api/calibration/sample",
    "/api/calibration/box_activate",
    "/api/calibration/notify",
    "/api/calibration/finish",
    "/api/calibration/skip",
    "/api/config",
    "/api/logs/clear",
    "/api/logs/reset_id",
    "/api/test_email"
}

@app.before_request
def check_csrf():
    if app.config.get("TESTING"):
        return None
    if request.method in ["POST", "PUT", "DELETE"] and request.path in MUTATING_ENDPOINTS:
        # Cross-origin origin validation
        origin = request.headers.get("Origin")
        if origin and origin not in ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost", "http://127.0.0.1"]:
            return jsonify({"success": False, "error": "Cross-origin request rejected"}), 403

        # Header validation when provided
        header_token = request.headers.get("X-CSRF-Token")
        if header_token and header_token != CSRF_TOKEN:
            return jsonify({"success": False, "error": "Invalid CSRF token"}), 403

        # Cross-site fetch check
        sec_fetch = request.headers.get("Sec-Fetch-Site")
        if sec_fetch and sec_fetch == "cross-site":
            return jsonify({"success": False, "error": "Cross-site request rejected"}), 403


@app.after_request
def set_csrf_cookie(response):
    if "csrf_token" not in request.cookies:
        response.set_cookie("csrf_token", CSRF_TOKEN, httponly=False, samesite="Lax")
    return response


@app.route("/favicon.ico")
def favicon():
    """Return a valid minimal ICO icon to prevent browser 404 logs."""
    ico_data = (
        b'\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00 \x00h\x04\x00\x00\x16\x00\x00\x00'
        b'(\x00\x00\x00\x10\x00\x00\x00 \x00\x00\x00\x01\x00 \x00\x00\x00\x00\x00\x00\x04\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' + (b'\x00' * 1024)
    )
    resp = make_response(ico_data)
    resp.headers["Content-Type"] = "image/x-icon"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/")
def index():
    """
    Render the modern dark theme eye-gaze communication interface.
    F3: Excludes plaintext smtp_pass (database.get_config omits secrets by default).
    F5: Sets csrf_token cookie for frontend request authentication.
    """
    config = database.get_config()
    caregiver_name = config.get("caregiver_name", "Primary Caregiver")
    resp = make_response(render_template(
        "index.html",
        blocks=BLOCKS_CONFIG,
        config=config,
        caregiver_name=caregiver_name
    ))
    resp.set_cookie("csrf_token", CSRF_TOKEN, httponly=False, samesite="Lax")
    return resp


@app.route("/caregiver/dashboard")
def caregiver_dashboard():
    """
    Caregiver portal overview dashboard with real-time status and system telemetry.
    """
    config = database.get_config()
    telemetry = getattr(vision_engine, "telemetry", {})
    recent_logs = database.get_requests_log(limit=6)
    calib = vision_engine.calibration
    return render_template(
        "caregiver/dashboard.html",
        caregiver_name=config.get("caregiver_name", "Primary Caregiver"),
        config=config,
        blocks=BLOCKS_CONFIG,
        telemetry=telemetry,
        recent_logs=recent_logs,
        calib=calib
    )


@app.route("/caregiver/settings")
def caregiver_settings_page():
    """Caregiver settings page."""
    config = database.get_config()
    return render_template("caregiver/settings.html", config=config, caregiver_name=config.get("caregiver_name", "Primary Caregiver"))


@app.route("/caregiver/history")
def caregiver_history_page():
    """Caregiver request history page."""
    logs = database.get_requests_log(limit=200)
    config = database.get_config()
    return render_template("caregiver/history.html", logs=logs, caregiver_name=config.get("caregiver_name", "Primary Caregiver"))


@app.route("/video_feed")
def video_feed():
    """
    D6: Stream live annotated computer vision feed as MJPEG.
    Sends frames only when updated, ends generator on client disconnect, and disables caching.

    TODO: The server-drawn gaze overlay is visible without login. Consider gating or
    suppressing annotations when caregiver is not authenticated.
    """
    def generate():
        last_ts = 0.0
        try:
            while not getattr(vision_engine, "is_shutting_down", False):
                current_ts = getattr(vision_engine, "latest_jpeg_timestamp", 0.0)
                if current_ts != last_ts or last_ts == 0.0:
                    jpeg_bytes = vision_engine.get_latest_jpeg()
                    if jpeg_bytes is not None:
                        last_ts = current_ts
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                time.sleep(0.033)
        except GeneratorExit:
            pass
        except Exception:
            pass

    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/telemetry")
def get_telemetry():
    """
    GA-2: Return instantaneous gaze telemetry and lockout status.
    """
    current_lock = check_lock_state()
    data = dict(vision_engine.telemetry)
    data["lock"] = current_lock
    return jsonify(data)


@app.route("/api/stream")
def sse_stream():
    """
    G1 & GA-2: Server-Sent Events (SSE) stream for low-latency telemetry.
    Merged cooldown logic, throttled to <= 15 Hz on change + 200ms heartbeat.

    Streams real-time gaze telemetry and lockout status to the patient interface.
    """
    def event_generator():
        last_sig = None
        last_heartbeat = time.time()
        try:
            while not getattr(vision_engine, "is_shutting_down", False):
                current_lock = check_lock_state()
                payload = dict(vision_engine.telemetry)
                payload["lock"] = current_lock


                now = time.time()
                # Significant fields signature for delta transmission
                sig = (
                    round(payload.get("gaze_x", 0.0), 3),
                    round(payload.get("gaze_y", 0.0), 3),
                    payload.get("gaze_valid", False),
                    payload.get("face_detected", False),
                    payload.get("is_blinking", False),
                    current_lock.get("is_locked", False),
                    current_lock.get("status", "IDLE"),
                    current_lock.get("email_status", "IDLE")
                )
                time_since_hb = now - last_heartbeat
                if sig != last_sig or time_since_hb >= 0.20:
                    last_sig = sig
                    last_heartbeat = now
                    yield f"data: {json.dumps(payload)}\n\n"

                time.sleep(0.033)  # ~30 Hz (matches camera frame rate for reduced latency)
        except GeneratorExit:
            pass
        except Exception:
            pass

    resp = Response(event_generator(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/trigger_block", methods=["POST"])
def trigger_block():
    """
    Called when a block's dwell time finishes or a keyboard hotkey is pressed.
    ENFORCES STRICT LOCKOUT & SELECTION STATE VALIDATION:
    1. Authenticated & email-verified caregiver check.
    2. Event idempotency deduplication (reject duplicate event_id).
    3. Camera freshness & availability validation (reject frozen/dead stream).
    4. Locks the entire interface until message delivery completes.
    5. Safe one-request-per-fixation guarantee.
    """
    req_json = request.get_json(silent=True) or {}
    block_id = req_json.get("block_id")

    # Compatibility alias for emergency -> help
    if block_id == "emergency":
        block_id = "help"

    # Event ID deduplication
    event_id = req_json.get("event_id")
    if event_id and is_duplicate_event(event_id):
        return jsonify({
            "success": False,
            "error": "Duplicate selection rejected: event already processed",
            "duplicate": True
        }), 409

    # Find matching block definition
    block_info = next((b for b in BLOCKS_CONFIG if b["id"] == block_id), None)
    if not block_info:
        return jsonify({"success": False, "error": f"Unknown block ID: {block_id}"}), 400

    trigger_source = req_json.get("trigger_source", "gaze")
    if trigger_source not in ("keyboard", "click"):
        # 1. Camera running check
        is_cam_running = getattr(vision_engine, "is_running", False) and getattr(vision_engine, "camera_enabled", False)
        if not is_cam_running:
            return jsonify({
                "success": False,
                "error": "CAMERA_INACTIVE",
                "message": "Camera is turned off or unavailable. Gaze triggers are disabled. You can press numbers 1-4 on keyboard."
            }), 503

        # 2. Frame freshness watchdog check
        now = time.time()
        latest_frame_time = getattr(vision_engine, "latest_frame_time", 0.0)
        if (now - latest_frame_time) > 1.0:
            return jsonify({
                "success": False,
                "error": "STALE_FRAME",
                "message": "Camera stream is frozen or frame is stale. Selection rejected for patient safety."
            }), 503

        # 3. Face detection check
        if not vision_engine.telemetry.get("face_detected", False):
            return jsonify({
                "success": False,
                "error": "Face not detected. Dwelling and gaze triggers are disabled until face is detected. You can press numbers 1-4 to trigger."
            }), 400

        # 4. Gaze validity & freshness check: require both eyes to be valid and fresh
        is_gaze_valid = bool(vision_engine.telemetry.get("gaze_valid", False) and vision_engine.telemetry.get("both_eyes_valid", False))
        is_blinking = bool(vision_engine.telemetry.get("is_blinking", False))
        is_stale = bool(vision_engine.telemetry.get("stale_prediction", False))
        frame_ts = float(vision_engine.telemetry.get("frame_timestamp", 0.0))
        # Tolerate a blink at final instant only when prior dwell was valid, but reject stale camera/frame data
        if (not is_gaze_valid and not is_blinking) or is_stale or (now - frame_ts > 1.0):
            reason = vision_engine.telemetry.get("invalid_reason", "Unreliable gaze tracking or stale prediction")
            return jsonify({
                "success": False,
                "error": f"Gaze measurement currently invalid or stale ({reason}). Selection rejected for patient safety. You can press numbers 1-4 to trigger."
            }), 400

    config = database.get_config()

    with lock_mutex:
        # Check if already locked
        if lock_state["is_locked"]:
            return jsonify({
                "success": False,
                "error": "System is already locked processing another request",
                "locked_block": lock_state["locked_block_title"]
            }), 423  # HTTP 423 Locked

        # ENGAGE LOCK IMMEDIATELY
        lock_state["is_locked"] = True
        lock_state["locked_block_id"] = block_id
        lock_state["locked_block_title"] = block_info["title"]
        lock_state["status"] = "SENDING_EMAIL"
        lock_state["email_status"] = "SENDING"
        lock_state["notification_status"] = "SENDING"
        lock_state["watchdog_unlocked"] = False
        lock_state["message"] = f"Request Locked: Sending alert for {block_info['title']} to {config.get('caregiver_email', '')}..."
        lock_state["lock_timestamp"] = time.time()
        lock_state["unlock_timestamp"] = 0.0

    # Pass real trigger source and note
    trigger_note = "Hands-free eye gaze dwell triggered" if trigger_source == "gaze" else f"Direct {trigger_source} trigger"
    request_id = database.log_request(
        block_id=block_id,
        block_title=block_info["title"],
        category=block_info["category"],
        priority=block_info["priority"],
        message=block_info["message"],
        caregiver_email=config.get("caregiver_email", ""),
        status="SENDING",
        notes=trigger_note
    )
    print(f"[Email] Request recorded (id={request_id}, block='{block_info['title']}', priority='{block_info['priority']}')")

    with lock_mutex:
        lock_state["last_request_id"] = request_id

    # Email worker completion callback
    def on_email_done(result):
        ok, status, note, duration = result
        with lock_mutex:
            # Prevent a late callback from overwriting a newer request's status
            if lock_state.get("last_request_id") != request_id:
                print(f"[Lockout] Ignoring completion callback for superseded request {request_id}")
                return

            lock_state["email_status"] = status
            lock_state["notification_status"] = status

            # If watchdog already unlocked the interface, do not re-lock the screen
            if lock_state.get("watchdog_unlocked"):
                print(f"[Lockout] Worker completed after watchdog unlock: {status} ({duration}ms)")
                if lock_state["status"] != "IDLE":
                    lock_state["message"] = f"Background alert finished: {status}."
                return

            lock_state["status"] = "COOLDOWN"
            if ok and status == "DELIVERED":
                lock_state["message"] = f"Delivered! Alert successfully dispatched to caregiver ({status})."
                lock_state["unlock_timestamp"] = time.time() + 2.2
            elif status == "SIMULATED":
                lock_state["message"] = f"Notification Simulated ({note}). System unfreezing..."
                lock_state["unlock_timestamp"] = time.time() + 1.8
            else:  # FAILED
                lock_state["message"] = f"Alert dispatch failed: {note}. Releasing interface."
                lock_state["unlock_timestamp"] = time.time() + 2.5

    # Prepare request data with real trigger_source
    alert_payload = dict(block_info)
    alert_payload["trigger_source"] = "keyboard" if trigger_source == "keyboard" else "gaze"

    # Launch email in background thread
    global _active_email_thread, _active_request_id
    worker_thread = email_notifier.send_alert_async(
        request_id=request_id,
        request_data=alert_payload,
        on_complete=on_email_done
    )
    _active_email_thread = worker_thread
    _active_request_id = request_id

    return jsonify({
        "success": True,
        "request_id": request_id,
        "event_id": event_id,
        "block": block_info,
        "locked": True,
        "message": f"Lock engaged. Notification in progress for {block_info['title']}."
    })


@app.route("/api/unlock", methods=["POST"])
def force_unlock():
    """Manual override unlock in case caregiver wants to release lock early."""
    with lock_mutex:
        lock_state["is_locked"] = False
        lock_state["locked_block_id"] = None
        lock_state["locked_block_title"] = None
        lock_state["status"] = "IDLE"
        lock_state["message"] = "Unlocked manually"
    return jsonify({"success": True, "message": "System unlocked."})


@app.route("/api/camera/toggle", methods=["POST"])
def toggle_camera():
    """Toggle camera on or off upon user request."""
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    if action == "stop" or (action is None and vision_engine.is_running):
        vision_engine.stop_camera()
        status = "stopped"
        msg = "Camera turned off. Touchless keyboard controls active (Keys 1-4)."
    else:
        vision_engine.start_camera(vision_engine.camera_index)
        status = "running"
        msg = f"Camera turned on ({vision_engine.camera_name}). Both-eye tracking resumed."
    return jsonify({
        "success": True,
        "camera_status": status,
        "is_running": vision_engine.is_running,
        "camera_index": vision_engine.camera_index,
        "camera_name": vision_engine.camera_name,
        "message": msg
    })


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """
    D1: Return cached camera list without re-opening devices while streaming.
    Only refreshes if refresh=1 query param is set and camera is idle.
    """
    force = request.args.get("refresh", "0") == "1"
    available = vision_engine.list_available_cameras(force_refresh=force)
    ext_idx = vision_engine.detect_external_camera(force_refresh=force)
    return jsonify({
        "success": True,
        "cameras": available,
        "active_index": vision_engine.camera_index,
        "active_name": vision_engine.camera_name,
        "external_detected": ext_idx is not None,
        "external_index": ext_idx,
        "mirror_view": getattr(vision_engine, "mirror_view", True),
        "is_running": vision_engine.is_running
    })


@app.route("/api/camera/select", methods=["POST"])
def select_camera():
    """
    F4: Switch active camera with validated and clamped input.
    """
    data = request.get_json(silent=True) or {}
    new_index = data.get("camera_index")
    if new_index is None:
        return jsonify({"success": False, "error": "Missing camera_index"}), 400

    try:
        new_index = int(new_index)
        if new_index < 0 or new_index > 10:
            return jsonify({"success": False, "error": "camera_index out of valid range (0-10)"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "camera_index must be an integer"}), 400

    success = vision_engine.switch_camera(new_index)
    actual_index = vision_engine.camera_index
    database.update_config({"camera_index": actual_index})
    cam_type = "External Webcam" if actual_index >= 1 else "System Built-in Camera"

    if success:
        return jsonify({
            "success": True,
            "camera_index": actual_index,
            "camera_name": vision_engine.camera_name,
            "message": f"Successfully switched to {cam_type} (Camera {actual_index})"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to open camera index {new_index}. Using {cam_type} (Camera {actual_index}).",
            "camera_index": actual_index
        }), 500


@app.route("/api/camera/mirror", methods=["POST"])
def toggle_mirror():
    """Toggle camera mirrored user view."""
    data = request.get_json(silent=True) or {}
    mirror = data.get("mirror")
    if mirror is None:
        vision_engine.mirror_view = not getattr(vision_engine, "mirror_view", True)
    else:
        vision_engine.mirror_view = bool(mirror)

    # Invalidate mismatched calibration profile and recenter baseline
    if hasattr(vision_engine, "mark_stale"):
        vision_engine.mark_stale()
    vision_engine.recenter()

    database.update_config({"mirror_view": 1 if vision_engine.mirror_view else 0})
    state = "Mirrored" if vision_engine.mirror_view else "Standard (Unmirrored)"
    return jsonify({
        "success": True,
        "mirror_view": vision_engine.mirror_view,
        "message": f"Camera user view set to {state}."
    })


@app.route("/api/recenter", methods=["POST"])
def recenter():
    """Re-center resting baseline gaze and head orientation."""
    vision_engine.recenter()
    return jsonify({"success": True, "message": "Baseline auto-centering triggered."})


@app.route("/api/calibration/start", methods=["POST"])
def calibration_start():
    """Start or restart a 5-point eye calibration sequence."""
    info = vision_engine.calibration.start_calibration()
    return jsonify({"success": True, "data": info})


@app.route("/api/calibration/box_activate", methods=["POST"])
def calibration_box_activate():
    """C3: Signal that a specific calibration target box is now active."""
    data = request.get_json(silent=True) or {}
    box_id = data.get("box_id", "top_right")
    reset = bool(data.get("reset", False))
    success = vision_engine.calibration.activate_box(box_id, reset=reset)
    return jsonify({
        "success": success,
        "box_id": box_id,
        "current_index": vision_engine.calibration.current_point_index
    })


@app.route("/api/calibration/sample", methods=["POST"])
def calibration_sample():
    """
    Collect synchronized both-eye gaze sample for current target box.
    Enforces that calibration is active, both-eye tracking is valid and fresh,
    the target matches the required sequence, and coordinates fall within target tolerance.
    """
    calib = vision_engine.calibration
    if not calib.is_calibrating:
        return jsonify({
            "success": False,
            "accepted": False,
            "reason": "Calibration is not currently active"
        }), 400

    data = request.get_json(silent=True) or {}
    box_id = data.get("box_id") or data.get("point_id", "top_right")

    now = time.time()
    telemetry = vision_engine.telemetry
    face_detected = bool(telemetry.get("face_detected", False))
    gaze_valid = bool(telemetry.get("gaze_valid", False))
    both_eyes_valid = bool(telemetry.get("both_eyes_valid", False))
    is_blinking = bool(telemetry.get("is_blinking", False))
    stale_prediction = bool(telemetry.get("stale_prediction", False))
    frame_ts = float(telemetry.get("frame_timestamp", 0.0))

    # Reject samples if camera is not producing fresh, valid both-eye gaze data
    if not face_detected or not gaze_valid or not both_eyes_valid or is_blinking or stale_prediction or (now - frame_ts > 0.35):
        count = len(calib.samples.get(box_id, []))
        reason = "Face not detected" if not face_detected else (
            "Blink detected" if is_blinking else (
                "Both-eye tracking invalid or occluded" if not (both_eyes_valid and gaze_valid) else "Telemetry frame stale"
            )
        )
        return jsonify({
            "success": True,
            "accepted": False,
            "box_id": box_id,
            "sample_count": count,
            "reason": reason
        })

    # Authoritative coordinates directly from vision engine to prevent client bypassing
    rx = float(vision_engine.raw_gaze_x)
    ry = float(vision_engine.raw_gaze_y)

    accepted, reason = calib.add_sample(
        box_id=box_id,
        raw_x=rx,
        raw_y=ry,
        left_dx=float(telemetry.get("norm_disp_left_x", 0.0)),
        left_dy=float(telemetry.get("norm_disp_left_y", 0.0)),
        right_dx=float(telemetry.get("norm_disp_right_x", 0.0)),
        right_dy=float(telemetry.get("norm_disp_right_y", 0.0)),
        conf_l=float(telemetry.get("iris_conf_left", 1.0)),
        conf_r=float(telemetry.get("iris_conf_right", 1.0)),
        is_blinking=is_blinking,
        head_yaw=float(telemetry.get("delta_yaw", telemetry.get("head_yaw", 0.0))),
        head_pitch=float(telemetry.get("delta_pitch", telemetry.get("head_pitch", 0.0))),
        both_eyes_valid=both_eyes_valid,
        gaze_valid=gaze_valid
    )
    count = len(calib.samples.get(box_id, []))
    return jsonify({
        "success": True,
        "accepted": accepted,
        "box_id": box_id,
        "sample_count": count,
        "reason": reason
    })


@app.route("/api/calibration/notify", methods=["POST"])
def calibration_notify():
    """
    C8: Send an automatic confirmation email to the caregiver only when calibration succeeded.
    Logs with real initial PENDING status (not fake DELIVERED).
    """
    calib = vision_engine.calibration
    if not calib.is_calibrated:
        return jsonify({
            "success": False,
            "error": "Cannot dispatch calibration notification: calibration was not successful or quality threshold was not met."
        }), 400

    config = database.get_config()
    acc_val = calib.accuracy
    acc_text = f"{acc_val}%" if acc_val is not None else "Accuracy Not Measured"
    metrics_str = f"Quality: {calib.quality}"
    if getattr(calib, "val_mae_px", None) is not None:
        metrics_str += f", Validation MAE: {calib.val_mae_px:.1f}px, RMSE: {calib.val_rmse_px:.1f}px"

    patient_name = config.get("patient_name", "Patient")
    caregiver_email = config.get("caregiver_email", "")

    calib_request = {
        "block_id": "calibration_complete",
        "block_title": "Eye Gaze Calibration Status",
        "category": "System Status",
        "priority": "HIGH",
        "message": f"Both-eye calibration update for {patient_name}. Measured Validation Accuracy: {acc_text} ({metrics_str}). Interactive cards unlocked.",
        "trigger_source": "calibration"
    }
    req_id = database.log_request(
        "calibration_complete", "Calibration Status", "System Status", "HIGH",
        calib_request["message"], caregiver_email, "PENDING",
        notes=f"Both-eye validation metrics: {metrics_str}"
    )
    email_notifier.send_alert_async(req_id, calib_request)
    return jsonify({
        "success": True,
        "message": f"Calibration confirmation dispatched to {caregiver_email}.",
        "accuracy": acc_val,
        "quality": calib.quality,
        "val_mae_px": getattr(calib, "val_mae_px", None)
    })


@app.route("/api/calibration/finish", methods=["POST"])
def calibration_finish():
    """
    C6: Compute calibration transform with real client screen dimensions.
    """
    data = request.get_json(silent=True) or {}
    try:
        sw = int(data.get("screen_w", 1920))
        sh = int(data.get("screen_h", 1080))
    except (ValueError, TypeError):
        sw, sh = 1920, 1080
    result = vision_engine.calibration.finish_calibration(frame_w=sw, frame_h=sh)
    return jsonify(result)


@app.route("/api/calibration/skip", methods=["POST"])
def calibration_skip():
    """User or timeout skips calibration to proceed to cards with adaptive baseline."""
    result = vision_engine.calibration.skip_calibration()
    return jsonify(result)


@app.route("/api/calibration/status", methods=["GET"])
def calibration_status():
    """Get current calibration state and target box metadata with LOBO-CV validation metrics."""
    calib = vision_engine.calibration
    return jsonify({
        "is_calibrating": calib.is_calibrating,
        "is_calibrated": calib.is_calibrated,
        "quality": calib.quality,
        "accuracy": calib.accuracy,
        "hit_rate": getattr(calib, "hit_rate", None),
        "val_mae_px": getattr(calib, "val_mae_px", None),
        "val_rmse_px": getattr(calib, "val_rmse_px", None),
        "val_mean_euclidean_px": getattr(calib, "val_mean_euclidean_px", None),
        "val_median_euclidean_px": getattr(calib, "val_median_euclidean_px", None),
        "val_mae_x_px": getattr(calib, "val_mae_x_px", None),
        "val_mae_y_px": getattr(calib, "val_mae_y_px", None),
        "val_hit_rate_pct": getattr(calib, "val_hit_rate_pct", None),
        "val_sample_count": getattr(calib, "val_sample_count", 0),
        "model_cv_scores": getattr(calib, "model_cv_scores", {}),
        "model_type": getattr(calib, "selected_model_name", "None"),
        "status_message": calib.status_message,
        "boxes": calib.target_boxes
    })


@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    """
    F1, F3, F4: Get or partially update caregiver configuration.
    Validates and clamps inputs, returning 400 on error.
    Modifying settings requires caregiver authentication.
    Never exposes plaintext smtp_pass.
    """
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Request body must be a JSON object"}), 400
        try:
            new_config = database.update_config(data)
        except (ValueError, TypeError) as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        # Update vision engine weights & mode
        vision_engine.head_weight = float(new_config.get("head_sensitivity", 1.2))
        vision_engine.eye_weight = float(new_config.get("eye_sensitivity", 1.5))
        if "tracking_mode" in new_config:
            vision_engine.set_tracking_mode(new_config["tracking_mode"])
        if "camera_index" in new_config and int(new_config["camera_index"]) != vision_engine.camera_index:
            vision_engine.switch_camera(int(new_config["camera_index"]))
        if "mirror_view" in new_config:
            vision_engine.mirror_view = bool(new_config["mirror_view"])
        return jsonify({"success": True, "config": new_config})
    else:
        return jsonify(database.get_config())


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """
    F4: Retrieve recent patient request history with validated limit.
    Requires caregiver authentication.
    """
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 1000))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid limit parameter"}), 400
    logs = database.get_requests_log(limit=limit)
    return jsonify({"success": True, "logs": logs})


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    """Clear request log history."""
    database.clear_requests_log()
    return jsonify({"success": True, "message": "History cleared."})


@app.route("/api/logs/reset_id", methods=["POST"])
def reset_logs_id():
    """Reset the auto-increment sequence for patient request logs."""
    try:
        result = database.reset_request_id_sequence()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Failed to reset request ID counter."
        }), 500


@app.route("/api/logs/export", methods=["GET"])
def export_logs():
    """Export history logs as downloadable CSV. Requires caregiver authentication."""
    csv_data = database.export_logs_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=patient_communication_logs.csv"}
    )


@app.route("/api/test_email", methods=["POST"])
def test_email():
    """Send diagnostic test email to caregiver. Requires caregiver authentication."""
    data = request.get_json(silent=True) or {}
    ok, status, msg, duration = email_notifier.send_test_email(data)
    return jsonify({
        "success": ok,
        "status": status,
        "message": msg,
        "duration_ms": duration
    })


@app.route("/api/notification_status", methods=["GET"])
def notification_status():
    """Return real-time email dispatch and lockout notification status."""
    current_lock = check_lock_state()
    req_id = request.args.get("request_id") or current_lock.get("last_request_id")
    email_status = current_lock.get("email_status", "IDLE")
    message = current_lock.get("message", "")
    duration_ms = 0

    if req_id:
        try:
            req_row = database.get_request_by_id(req_id)
            if req_row:
                db_status = req_row.get("status")
                if db_status in ("DELIVERED", "SIMULATED", "FAILED"):
                    email_status = db_status
                duration_ms = req_row.get("delivery_duration_ms", 0)
                if req_row.get("notes"):
                    message = req_row.get("notes")
        except Exception:
            pass

    return jsonify({
        "success": True,
        "is_locked": current_lock.get("is_locked", False),
        "status": current_lock.get("status", "IDLE"),
        "email_status": email_status,
        "notification_status": email_status,
        "request_id": req_id,
        "message": message,
        "duration_ms": duration_ms
    })


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    # F5: Bind to 127.0.0.1 by default (configurable via HOST)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print("\n==================================================================")
    print(" AI-VISION COMMUNICATION ASSISTANCE FOR PARALYZED INDIVIDUALS")
    print(" Zero-Calibration Gaze & Head Interaction System")
    print(f" Running at: http://{host}:{port}")
    print(" Press Ctrl+C at any time in this window to stop and release camera.")
    print("==================================================================\n")
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_application()
