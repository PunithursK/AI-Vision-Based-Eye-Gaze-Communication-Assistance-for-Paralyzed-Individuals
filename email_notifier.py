"""
Caregiver Notification and Automated Email Dispatcher.
Uses Brevo Transactional Email REST API (https://api.brevo.com/v3/smtp/email)
with professional HTML alerts, asynchronous dispatch, and offline simulation mode.
"""

import os
import requests
import threading
import time
import html
import re
import logging
from datetime import datetime
import database

logger = logging.getLogger("email_notifier")

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

# Thread-safe in-flight request deduplication
_in_flight_lock = threading.Lock()
_in_flight_requests = set()


def _is_valid_email(addr):
    """Check basic email address format."""
    if not addr or not isinstance(addr, str):
        return False
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", addr.strip()))


def _sanitize_error_message(err, secret=""):
    """Format and sanitize exception message to avoid exposing API keys or sensitive details."""
    if not err:
        return "Unknown error"
    err_str = f"{type(err).__name__}: {str(err)}" if str(err) else type(err).__name__
    if secret and str(secret).strip():
        err_str = err_str.replace(str(secret).strip(), "******")
    return err_str


def prewarm_connection():
    """
    Callable interface preserved for existing callers.
    With Brevo REST API, prewarming is a lightweight no-op.
    """
    return None


def build_email_html(request_data, patient_name):
    """Generate a high-visibility, professional HTML email alert with escaped user content."""
    priority = html.escape(str(request_data.get("priority", "MEDIUM")).upper())
    block_title = html.escape(str(request_data.get("block_title", "Request")))
    message = html.escape(str(request_data.get("message", "")))
    category = html.escape(str(request_data.get("category", "General")))
    patient_name_esc = html.escape(str(patient_name))
    trigger_source = html.escape(str(request_data.get("trigger_source", "Hands-Free Eye & Head Gaze (AI Vision)")))
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    # Priority styling
    if priority == "CRITICAL":
        badge_bg = "#dc2626"
        badge_text = "CRITICAL EMERGENCY"
        header_color = "#991b1b"
        accent_color = "#ef4444"
    elif priority == "HIGH":
        badge_bg = "#ea580c"
        badge_text = "HIGH URGENCY"
        header_color = "#c2410c"
        accent_color = "#f97316"
    else:
        badge_bg = "#0284c7"
        badge_text = "STANDARD ASSISTANCE"
        header_color = "#0369a1"
        accent_color = "#38bdf8"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #0b0f19;
                color: #e2e8f0;
                margin: 0;
                padding: 24px;
            }}
            .card {{
                max-width: 600px;
                margin: 0 auto;
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            }}
            .header {{
                background: {header_color};
                padding: 24px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                color: #ffffff;
                font-size: 22px;
                letter-spacing: 0.5px;
            }}
            .badge {{
                display: inline-block;
                background-color: {badge_bg};
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                padding: 4px 12px;
                border-radius: 9999px;
                margin-top: 8px;
                letter-spacing: 1px;
            }}
            .body-content {{
                padding: 28px;
            }}
            .alert-box {{
                background: #1e293b;
                border-left: 4px solid {accent_color};
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 24px;
            }}
            .alert-title {{
                font-size: 20px;
                font-weight: 700;
                color: #ffffff;
                margin: 0 0 8px 0;
            }}
            .alert-message {{
                font-size: 16px;
                color: #94a3b8;
                line-height: 1.5;
            }}
            .meta-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 24px;
            }}
            .meta-table tr {{
                border-bottom: 1px solid #1f2937;
            }}
            .meta-table td {{
                padding: 10px 0;
                font-size: 14px;
            }}
            .meta-label {{
                color: #64748b;
                width: 38%;
            }}
            .meta-value {{
                color: #e2e8f0;
                font-weight: 500;
            }}
            .action-box {{
                background: rgba(30, 41, 59, 0.5);
                border: 1px dashed #334155;
                border-radius: 8px;
                padding: 14px 16px;
                font-size: 13px;
                color: #94a3b8;
                text-align: center;
            }}
            .footer {{
                background: #0d1321;
                padding: 16px;
                text-align: center;
                font-size: 11px;
                color: #475569;
                border-top: 1px solid #1f2937;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h1>{badge_text}</h1>
                <div class="badge">{priority} PRIORITY</div>
            </div>
            <div class="body-content">
                <div class="alert-box">
                    <div class="alert-title">{block_title}</div>
                    <div class="alert-message">{message}</div>
                </div>

                <table class="meta-table">
                    <tr>
                        <td class="meta-label">Patient Identity:</td>
                        <td class="meta-value">{patient_name_esc}</td>
                    </tr>
                    <tr>
                        <td class="meta-label">Category:</td>
                        <td class="meta-value">{category}</td>
                    </tr>
                    <tr>
                        <td class="meta-label">Priority Level:</td>
                        <td class="meta-value">{priority}</td>
                    </tr>
                    <tr>
                        <td class="meta-label">Detection Method:</td>
                        <td class="meta-value">{trigger_source}</td>
                    </tr>
                    <tr>
                        <td class="meta-label">Timestamp:</td>
                        <td class="meta-value">{timestamp}</td>
                    </tr>
                </table>

                <div class="action-box">
                    <strong>Caregiver Action Required:</strong> Please attend to the patient promptly or verify their condition.
                </div>
            </div>
            <div class="footer">
                AI-Vision-Based Communication Assistance for Paralyzed Individuals<br>
                Automated notification generated by hands-free gaze tracking system via Brevo API.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


def send_email_sync(request_id, request_data, config=None):
    """
    Synchronously attempt to send email via Brevo Transactional Email REST API or simulate delivery.
    Returns (success: bool, status: str, message: str, duration_ms: int).
    """
    start_time = time.time()
    block_title = request_data.get("block_title", "Assistance Request")
    print(f"[Email] Dispatch started for request #{request_id} ({block_title}) via Brevo API")

    stored_cfg = database.get_email_credentials()
    if config is None:
        cfg = stored_cfg
    else:
        cfg = dict(stored_cfg)
        cfg.update(config)
        # If masked placeholder was passed, keep stored key
        if "brevo_api_key" in config and "•" in str(config["brevo_api_key"]):
            cfg["brevo_api_key"] = stored_cfg.get("brevo_api_key") or ""

    recipient_email = (cfg.get("caregiver_email") or "").strip()
    caregiver_name = (cfg.get("caregiver_name") or "Caregiver").strip()
    patient_name = (cfg.get("patient_name") or "Patient").strip()
    brevo_sender_name = str(cfg.get("brevo_sender_name") if cfg.get("brevo_sender_name") is not None else "Eye Gaze AI").strip()
    brevo_sender_email = str(cfg.get("brevo_sender_email") if cfg.get("brevo_sender_email") is not None else "ogjoyboy21@gmail.com").strip()
    sim_mode = bool(cfg.get("simulation_mode", 0))

    # Brevo API Key resolution (env var takes precedence)
    raw_api_key = cfg.get("brevo_api_key") or ""
    brevo_api_key = str(raw_api_key).strip()

    env_key = os.environ.get("BREVO_API_KEY")
    if env_key and env_key.strip():
        brevo_api_key = env_key.strip()

    priority = request_data.get("priority", "MEDIUM")
    subject = f"[{priority}] Patient Alert: {request_data.get('block_title', 'Assistance Request')} - {patient_name}"

    # 1. Recipient validation: strictly required for actual delivery
    if not recipient_email or not _is_valid_email(recipient_email):
        duration_ms = int((time.time() - start_time) * 1000)
        err = "Caregiver recipient email is missing or invalid. Please configure a valid caregiver alert recipient in Settings."
        database.update_request_status(request_id, "FAILED", duration_ms, err)
        print(f"[Email] Validation failed: missing/invalid recipient (Status: FAILED)")
        return False, "FAILED", err, duration_ms

    # 2. Simulation mode: default-safe offline simulation
    if sim_mode:
        time.sleep(0.08)
        duration_ms = int((time.time() - start_time) * 1000)
        status_note = f"Simulated delivery to {recipient_email}. (Simulation mode — no real email is sent)"
        database.update_request_status(request_id, "SIMULATED", duration_ms, status_note)
        print(f"[Email] Dispatch completed in {duration_ms} ms (Status: SIMULATED)")
        return True, "SIMULATED", "Simulation mode — no real email is sent", duration_ms

    # 3. Live Brevo mode preflight validation: API Key
    if not brevo_api_key:
        duration_ms = int((time.time() - start_time) * 1000)
        err = "Brevo API key is missing. Please configure your Brevo API key in Settings or set BREVO_API_KEY environment variable."
        database.update_request_status(request_id, "FAILED", duration_ms, err)
        print(f"[Email] Validation failed: missing Brevo API key (Status: FAILED)")
        return False, "FAILED", err, duration_ms

    # 4. Live Brevo mode preflight validation: Sender Email
    if not brevo_sender_email or not _is_valid_email(brevo_sender_email):
        duration_ms = int((time.time() - start_time) * 1000)
        err = "Brevo sender email is missing or invalid. Please configure a valid sender email in Settings."
        database.update_request_status(request_id, "FAILED", duration_ms, err)
        print(f"[Email] Validation failed: missing/invalid sender email (Status: FAILED)")
        return False, "FAILED", err, duration_ms

    # Truthful status transition: mark SENDING while request is processing
    database.update_request_status(request_id, "SENDING", 0, f"Dispatching alert via Brevo API to {recipient_email}...")

    html_body = build_email_html(request_data, patient_name)
    plain_body = f"[{priority}] Alert from {patient_name}:\n{request_data.get('block_title')}: {request_data.get('message')}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    payload = {
        "sender": {
            "name": brevo_sender_name,
            "email": brevo_sender_email
        },
        "to": [
            {
                "email": recipient_email,
                "name": caregiver_name
            }
        ],
        "subject": subject,
        "htmlContent": html_body,
        "textContent": plain_body
    }

    headers = {
        "accept": "application/json",
        "api-key": brevo_api_key,
        "content-type": "application/json"
    }

    try:
        t_post = time.time()
        resp = requests.post(
            BREVO_API_URL,
            json=payload,
            headers=headers,
            timeout=8.0
        )
        post_ms = int((time.time() - t_post) * 1000)
        duration_ms = int((time.time() - start_time) * 1000)

        # Interpret HTTP 201 as accepted by Brevo, NOT proof of inbox receipt
        if resp.status_code == 201:
            msg_id = ""
            try:
                resp_data = resp.json()
                msg_id = resp_data.get("messageId", "")
            except Exception:
                pass

            status_note = f"Accepted by Brevo API (201 Created) for delivery to {recipient_email} in {post_ms}ms"
            if msg_id:
                status_note += f" (Message ID: {msg_id})"

            print(f"[Email] Brevo API accepted message in {post_ms} ms (Status: 201 Created)")
            print(f"[Email] Dispatch completed in {duration_ms} ms (Status: DELIVERED)")
            database.update_request_status(request_id, "DELIVERED", duration_ms, status_note)
            return True, "DELIVERED", status_note, duration_ms

        else:
            # Map non-201 HTTP status codes to actionable, sanitized messages
            raw_msg = ""
            try:
                err_json = resp.json()
                raw_msg = err_json.get("message", "")
            except Exception:
                raw_msg = resp.text[:200]

            if resp.status_code == 401:
                err_msg = "Brevo authentication failed (401 Unauthorized): Invalid or inactive Brevo API key. Please check your Brevo account."
            elif resp.status_code == 403:
                err_msg = f"Brevo permission denied (403 Forbidden): Sender email '{brevo_sender_email}' is unverified, or account is unactivated in Brevo."
            elif resp.status_code == 400:
                err_msg = f"Brevo rejected request (400 Bad Request): {raw_msg}."
            elif resp.status_code == 429:
                err_msg = "Brevo rate limit exceeded (429 Too Many Requests). Please retry shortly."
            elif resp.status_code >= 500:
                err_msg = f"Brevo API server error ({resp.status_code}). Please retry shortly."
            else:
                err_msg = f"Brevo API error ({resp.status_code}): {raw_msg}"

            err_msg = _sanitize_error_message(err_msg, brevo_api_key)
            print(f"[Email] Brevo API returned {resp.status_code} in {post_ms} ms (Status: FAILED)")
            print(f"[Email] Dispatch completed in {duration_ms} ms (Status: FAILED)")
            database.update_request_status(request_id, "FAILED", duration_ms, err_msg)
            return False, "FAILED", err_msg, duration_ms

    except requests.exceptions.Timeout:
        duration_ms = int((time.time() - start_time) * 1000)
        err_msg = "Connection to Brevo API timed out (server unreachable or network delayed)."
        print(f"[Email] Brevo API request timed out (Status: FAILED)")
        database.update_request_status(request_id, "FAILED", duration_ms, err_msg)
        return False, "FAILED", err_msg, duration_ms

    except requests.exceptions.ConnectionError:
        duration_ms = int((time.time() - start_time) * 1000)
        err_msg = "Could not connect to Brevo API (network offline or DNS resolution failed)."
        print(f"[Email] Brevo API connection error (Status: FAILED)")
        database.update_request_status(request_id, "FAILED", duration_ms, err_msg)
        return False, "FAILED", err_msg, duration_ms

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        err_msg = _sanitize_error_message(e, brevo_api_key)
        print(f"[Email] Brevo API dispatch exception: {err_msg} (Status: FAILED)")
        database.update_request_status(request_id, "FAILED", duration_ms, err_msg)
        return False, "FAILED", err_msg, duration_ms


def send_alert_async(request_id, request_data, on_complete=None):
    """
    Launch email dispatch in a background thread to prevent UI lockup.
    Always invokes on_complete(result) even if worker encounters an unhandled exception.
    Includes thread-safe in-flight deduplication.
    """
    with _in_flight_lock:
        if request_id in _in_flight_requests:
            logger.warning(f"[Email] Request #{request_id} is already in-flight. Skipping duplicate dispatch.")
            return None
        _in_flight_requests.add(request_id)

    start_time = time.time()

    def _worker():
        try:
            res = send_email_sync(request_id, request_data)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            res = (False, "FAILED", f"Worker exception: {_sanitize_error_message(e)}", duration_ms)
        finally:
            with _in_flight_lock:
                _in_flight_requests.discard(request_id)

        if on_complete:
            try:
                on_complete(res)
            except Exception as e:
                logger.debug(f"[Notifier Callback Error] {e}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def send_test_email(config_override=None):
    """
    Send a diagnostic test email to verify Brevo API configuration.
    Reports whether it was simulated, rejected by missing configuration, accepted by Brevo, or failed.
    """
    stored_cfg = database.get_email_credentials()
    if config_override is None:
        cfg = stored_cfg
    else:
        cfg = dict(stored_cfg)
        for k, v in config_override.items():
            if k == "brevo_api_key":
                if v and "•" not in str(v):
                    cfg[k] = str(v).strip()
            else:
                cfg[k] = v

    recipient_email = (cfg.get("caregiver_email") or "").strip()
    test_request = {
        "block_id": "test",
        "block_title": "System Connectivity Test",
        "category": "Diagnostics",
        "priority": "MEDIUM",
        "message": "This is a diagnostic verification email confirming that the AI-Vision Communication Assistance system can successfully alert caregivers via Brevo Transactional Email API.",
        "trigger_source": "System Connectivity Diagnostic"
    }
    req_id = database.log_request(
        "test", "System Test", "Diagnostics", "MEDIUM",
        test_request["message"], recipient_email, "SENDING"
    )
    return send_email_sync(req_id, test_request, cfg)
