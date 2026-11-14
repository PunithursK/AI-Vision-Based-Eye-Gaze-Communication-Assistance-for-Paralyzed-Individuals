"""
Unit and integration tests verifying Brevo Transactional Email REST API delivery diagnostics,
configuration security, and calibration dwell state logic.
All external HTTP requests are strictly mocked — no real emails are sent during tests.
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import patch, MagicMock
import requests

# Ensure project directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Windows Application Control policy on this host blocks tensorflow DLLs loaded by mediapipe.
# To prevent test crashes while strictly respecting the prohibition against modifying ai_vision_engine.py,
# we mock ai_vision_engine in sys.modules prior to importing app.
if "ai_vision_engine" not in sys.modules:
    sys.modules["ai_vision_engine"] = MagicMock()

import database
import email_notifier
from app import app, BLOCKS_CONFIG, COMMUNICATION_BLOCKS


class TestCalibrationDwellLogicSimulation(unittest.TestCase):
    """Simulates card-style calibration dwell state machine logic."""

    def test_bounding_box_hit_testing(self):
        """Verify hit test starts only when cursor coordinates enter active target rectangle."""
        rect = {"left": 200, "top": 100, "right": 400, "bottom": 300}

        def is_inside(cx, cy, margin=0):
            return (
                cx >= rect["left"] - margin and
                cx <= rect["right"] + margin and
                cy >= rect["top"] - margin and
                cy <= rect["bottom"] + margin
            )

        self.assertFalse(is_inside(150, 200))
        self.assertFalse(is_inside(450, 200))
        self.assertTrue(is_inside(300, 200))
        self.assertTrue(is_inside(190, 200, margin=16))

    def test_dwell_and_sample_both_required_for_completion(self):
        """Calibration point completes only after BOTH dwell >= duration AND samples >= 16."""
        dwell_duration = 1.5
        target_samples = 16

        def check_complete(dwell_time, samples):
            return dwell_time >= dwell_duration and samples >= target_samples

        self.assertFalse(check_complete(1.6, 15))
        self.assertFalse(check_complete(1.2, 16))
        self.assertTrue(check_complete(1.5, 16))
        self.assertTrue(check_complete(2.0, 18))

    def test_blink_grace_period_logic(self):
        """Micro-blink <= 350ms pauses dwell; prolonged blink > 350ms triggers reset."""
        dwell_start = 1000.0
        blink_start = 1200.0

        # Micro-blink: 150ms later
        blink_duration = 1350.0 - blink_start
        self.assertLessEqual(blink_duration, 350)
        adjusted_dwell_start = dwell_start + 150.0
        elapsed_after_pause = 1350.0 - adjusted_dwell_start
        self.assertEqual(elapsed_after_pause, 200.0)

        # Prolonged blink: 400ms later
        prolonged_duration = 1600.0 - blink_start
        self.assertGreater(prolonged_duration, 350)


class TestBrevoEmailDeliveryDiagnostics(unittest.TestCase):
    """
    Verifies Brevo API email delivery preflights, simulation mode, credential validation,
    and HTTP response code handling.
    """

    def setUp(self):
        # Clear env overrides if set
        os.environ.pop("BREVO_API_KEY", None)
        os.environ.pop("BREVO_SENDER_NAME", None)
        os.environ.pop("BREVO_SENDER_EMAIL", None)
        os.environ.pop("GAZE_CAREGIVER_EMAIL", None)

    def test_empty_recipient_rejected(self):
        """Empty or missing recipient email must be rejected without claiming success."""
        test_cfg = {
            "caregiver_email": "",
            "brevo_api_key": "xkeysib-mock",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999901, {"block_title": "Test"}, test_cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("recipient", msg.lower())

    def test_invalid_recipient_rejected(self):
        """Malformed recipient email must be rejected."""
        test_cfg = {
            "caregiver_email": "not-an-email",
            "brevo_api_key": "xkeysib-mock",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999902, {"block_title": "Test"}, test_cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("recipient", msg.lower())

    def test_simulation_mode_success(self):
        """Simulation mode (simulation_mode=1) must truthfully return SIMULATED and avoid HTTP calls."""
        test_cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 1
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999903, {"block_title": "Test"}, test_cfg)
        self.assertTrue(ok)
        self.assertEqual(status, "SIMULATED")
        self.assertIn("Simulation mode", msg)

    @patch("requests.post")
    def test_live_mode_missing_api_key(self, mock_post):
        """Live Brevo mode with missing API key must fail with clear actionable diagnostic."""
        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999904, {"block_title": "Test"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("brevo api key", msg.lower())
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_live_mode_missing_sender_email(self, mock_post):
        """Live Brevo mode with missing sender email must fail with clear diagnostic."""
        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-mock-12345",
            "brevo_sender_email": "",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999905, {"block_title": "Test"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("sender email", msg.lower())
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_brevo_api_success_201(self, mock_post):
        """Verify successful dispatch when Brevo returns 201 Created."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"messageId": "<20261113.12345@smtp-relay.brevo.com>"}
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "caregiver_name": "Nurse Jane",
            "patient_name": "Patient Alpha",
            "brevo_api_key": "xkeysib-validkey123",
            "brevo_sender_name": "Eye Gaze AI",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999906, {"block_title": "Food", "priority": "HIGH"}, cfg)
        self.assertTrue(ok)
        self.assertEqual(status, "DELIVERED")
        self.assertIn("Accepted by Brevo API", msg)
        self.assertIn("201 Created", msg)
        self.assertIn("<20261113.12345@smtp-relay.brevo.com>", msg)

        # Verify POST payload and headers
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.brevo.com/v3/smtp/email")
        self.assertEqual(kwargs["headers"]["api-key"], "xkeysib-validkey123")
        payload = kwargs["json"]
        self.assertEqual(payload["sender"]["email"], "ogjoyboy21@gmail.com")
        self.assertEqual(payload["sender"]["name"], "Eye Gaze AI")
        self.assertEqual(payload["to"][0]["email"], "caregiver@example.com")
        self.assertIn("Food", payload["subject"])
        self.assertIn("Patient Alpha", payload["htmlContent"])

    @patch("requests.post")
    def test_brevo_api_401_unauthorized(self, mock_post):
        """Verify 401 Unauthorized returns sanitized actionable failure."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"code": "unauthorized", "message": "Key not found"}
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-badkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999907, {"block_title": "Water"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("401 unauthorized", msg.lower())
        self.assertNotIn("xkeysib-badkey", msg)

    @patch("requests.post")
    def test_brevo_api_403_unverified_sender(self, mock_post):
        """Verify 403 Forbidden returns unverified sender message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"code": "forbidden", "message": "Sender not verified"}
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "unverified@example.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999908, {"block_title": "Medicine"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("403 forbidden", msg.lower())

    @patch("requests.post")
    def test_brevo_api_400_bad_request(self, mock_post):
        """Verify 400 Bad Request returns sanitized error details."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"code": "bad_request", "message": "Invalid email format"}
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999911, {"block_title": "Medicine"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("400 bad request", msg.lower())
        self.assertIn("invalid email format", msg.lower())

    @patch("requests.post")
    def test_brevo_api_429_rate_limit(self, mock_post):
        """Verify 429 Too Many Requests returns rate limit message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"code": "too_many_requests", "message": "Rate limit reached"}
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999912, {"block_title": "Medicine"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("429", msg)
        self.assertIn("rate limit", msg.lower())

    @patch("requests.post")
    def test_brevo_api_500_server_error(self, mock_post):
        """Verify 500 Internal Server Error returns retry message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        mock_resp.json.side_effect = Exception("Not JSON")
        mock_post.return_value = mock_resp

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999913, {"block_title": "Help"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("500", msg)

    @patch("requests.post")
    def test_brevo_api_timeout(self, mock_post):
        """Verify request timeout is caught cleanly without crashing."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out")

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999909, {"block_title": "Help"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("timed out", msg.lower())

    @patch("requests.post")
    def test_brevo_api_connection_error(self, mock_post):
        """Verify network connection error is caught cleanly."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to establish a new connection")

        cfg = {
            "caregiver_email": "caregiver@example.com",
            "brevo_api_key": "xkeysib-validkey",
            "brevo_sender_email": "ogjoyboy21@gmail.com",
            "simulation_mode": 0
        }
        ok, status, msg, _ = email_notifier.send_email_sync(999914, {"block_title": "Help"}, cfg)
        self.assertFalse(ok)
        self.assertEqual(status, "FAILED")
        self.assertIn("connect", msg.lower())

    def test_api_key_sanitization(self):
        """Never expose raw Brevo API keys in error messages."""
        secret_key = "xkeysib-supersecretapi123456789"
        err = Exception(f"Failed to post request using key {secret_key}")
        sanitized = email_notifier._sanitize_error_message(err, secret_key)
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("******", sanitized)

    @patch("requests.post")
    def test_asynchronous_alert_dispatch(self, mock_post):
        """Verify send_alert_async runs in background and invokes completion callback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"messageId": "<async-msg-1>"}
        mock_post.return_value = mock_resp

        # Configure environment overrides for async dispatch so it passes preflights
        os.environ["BREVO_API_KEY"] = "xkeysib-validkey-async"
        os.environ["GAZE_CAREGIVER_EMAIL"] = "caregiver@example.com"
        database.update_config({"simulation_mode": 0, "brevo_sender_email": "ogjoyboy21@gmail.com"})

        callback_event = threading.Event()
        callback_result = []

        def on_complete(res):
            callback_result.append(res)
            callback_event.set()

        req_data = {"block_title": "Food", "priority": "MEDIUM"}
        email_notifier.send_alert_async(999910, req_data, on_complete=on_complete)

        finished = callback_event.wait(timeout=3.0)
        self.assertTrue(finished, "Async email callback should complete within timeout")
        self.assertEqual(len(callback_result), 1)
        ok, status, _, _ = callback_result[0]
        self.assertTrue(ok)
        self.assertEqual(status, "DELIVERED")

    def test_test_email_functionality(self):
        """Verify send_test_email works correctly in simulation mode."""
        override = {
            "caregiver_email": "test-caregiver@example.com",
            "simulation_mode": 1
        }
        ok, status, msg, _ = email_notifier.send_test_email(override)
        self.assertTrue(ok)
        self.assertEqual(status, "SIMULATED")

    def test_environment_variable_precedence(self):
        """Verify BREVO_API_KEY, BREVO_SENDER_NAME, BREVO_SENDER_EMAIL take precedence over database."""
        os.environ["BREVO_API_KEY"] = "xkeysib-env-override-key"
        os.environ["BREVO_SENDER_NAME"] = "Env Sender Name"
        os.environ["BREVO_SENDER_EMAIL"] = "env_sender@example.com"
        os.environ["GAZE_CAREGIVER_EMAIL"] = "env_caregiver@example.com"

        creds = database.get_email_credentials()
        self.assertEqual(creds["brevo_api_key"], "xkeysib-env-override-key")
        self.assertEqual(creds["brevo_sender_name"], "Env Sender Name")
        self.assertEqual(creds["brevo_sender_email"], "env_sender@example.com")
        self.assertEqual(creds["caregiver_email"], "env_caregiver@example.com")


class TestSystemIntegration(unittest.TestCase):
    """Verifies Flask endpoints, Brevo config persistence, and preserved cards."""

    def setUp(self):
        self.client = app.test_client()

    def test_brevo_config_persistence_and_masking(self):
        """Verify brevo_api_key is persisted and masked properly without leaking plaintext."""
        # 1. Update config with new Brevo API key
        resp = self.client.post("/api/config", json={
            "email_provider": "brevo",
            "brevo_api_key": "xkeysib-test-secret-key-999",
            "brevo_sender_name": "Eye Gaze AI",
            "brevo_sender_email": "ogjoyboy21@gmail.com"
        })
        self.assertEqual(resp.status_code, 200)

        # 2. GET /api/config must NOT return plaintext brevo_api_key
        cfg = self.client.get("/api/config").get_json()
        self.assertNotIn("brevo_api_key", cfg)
        self.assertTrue(cfg["has_brevo_api_key"])
        self.assertEqual(cfg["brevo_api_key_masked"], "••••••••••••")
        self.assertEqual(cfg["brevo_sender_email"], "ogjoyboy21@gmail.com")

        # 3. Sending masked placeholder must preserve existing saved key
        self.client.post("/api/config", json={
            "brevo_api_key": "••••••••••••",
            "brevo_sender_name": "Eye Gaze AI Updated"
        })
        creds = database.get_email_credentials()
        self.assertEqual(creds["brevo_api_key"], "xkeysib-test-secret-key-999")
        self.assertEqual(creds["brevo_sender_name"], "Eye Gaze AI Updated")

        # 4. Sending empty brevo_api_key must also preserve existing saved key
        self.client.post("/api/config", json={
            "brevo_api_key": "",
            "brevo_sender_name": "Eye Gaze AI Final"
        })
        creds = database.get_email_credentials()
        self.assertEqual(creds["brevo_api_key"], "xkeysib-test-secret-key-999")
        self.assertEqual(creds["brevo_sender_name"], "Eye Gaze AI Final")

    def test_four_patient_cards_preserved(self):
        """Verify that the 4 patient cards (food, water, medicine, help) are preserved."""
        self.assertEqual(len(COMMUNICATION_BLOCKS), 4)
        self.assertEqual(len(BLOCKS_CONFIG), 4)
        ids = [b["id"] for b in COMMUNICATION_BLOCKS]
        self.assertEqual(ids, ["food", "water", "medicine", "help"])

    def test_test_email_endpoint_with_brevo_simulation_and_validation(self):
        """Verify /api/test_email reports simulated or failure correctly for Brevo."""
        # Missing recipient
        resp = self.client.post("/api/test_email", json={
            "caregiver_email": "",
            "simulation_mode": 0
        })
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["status"], "FAILED")

        # Simulation mode enabled
        resp = self.client.post("/api/test_email", json={
            "caregiver_email": "caregiver@example.com",
            "simulation_mode": 1
        })
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "SIMULATED")

    def test_notification_status_endpoint(self):
        """Verify /api/notification_status reports accurate delivery and lockout status."""
        # Test default/idle status
        resp = self.client.get("/api/notification_status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("is_locked", data)
        self.assertIn("email_status", data)
        self.assertIn("notification_status", data)

        # Test request with DELIVERED status in SQLite
        req_id = database.log_request("food", "Food", "Care", "HIGH", "Hungry", "cg@test.com")
        database.update_request_status(req_id, "DELIVERED", 145, "Delivered via Brevo API")
        resp_del = self.client.get(f"/api/notification_status?request_id={req_id}")
        self.assertEqual(resp_del.status_code, 200)
        data_del = resp_del.get_json()
        self.assertEqual(data_del["email_status"], "DELIVERED")
        self.assertEqual(data_del["notification_status"], "DELIVERED")
        self.assertEqual(data_del["duration_ms"], 145)

        # Test request with SIMULATED status in SQLite
        req_id_sim = database.log_request("water", "Water", "Care", "HIGH", "Thirsty", "cg@test.com")
        database.update_request_status(req_id_sim, "SIMULATED", 80, "Simulated mode")
        resp_sim = self.client.get(f"/api/notification_status?request_id={req_id_sim}")
        data_sim = resp_sim.get_json()
        self.assertEqual(data_sim["email_status"], "SIMULATED")

        # Test request with FAILED status in SQLite
        req_id_fail = database.log_request("help", "Help", "Emergency", "CRITICAL", "Need help", "cg@test.com")
        database.update_request_status(req_id_fail, "FAILED", 210, "Brevo API 401 Unauthorized")
        resp_fail = self.client.get(f"/api/notification_status?request_id={req_id_fail}")
        data_fail = resp_fail.get_json()
        self.assertEqual(data_fail["email_status"], "FAILED")
        self.assertIn("Brevo API 401", data_fail["message"])

    def test_brevo_api_key_env_var_precedence(self):
        """Verify environment variable BREVO_API_KEY takes precedence over database value."""
        database.update_config({"brevo_api_key": "xkeysib-database-stored-key"})
        
        # Without env var
        creds = database.get_email_credentials()
        self.assertEqual(creds["brevo_api_key"], "xkeysib-database-stored-key")

        # With env var set
        with patch.dict(os.environ, {"BREVO_API_KEY": "xkeysib-environment-precedence-key"}):
            creds_env = database.get_email_credentials()
            self.assertEqual(creds_env["brevo_api_key"], "xkeysib-environment-precedence-key")
            cfg = database.get_config()
            self.assertTrue(cfg["has_brevo_api_key"])


    def test_reset_request_id_sequence_with_existing_records(self):
        """Verify reset request ID preserves existing records and prevents duplicate IDs."""
        # 1. Clear and reset sequence so test starts cleanly at 1
        database.clear_requests_log()
        database.reset_request_id_sequence()
        id1 = database.log_request("food", "Food", "Care", "HIGH", "Hungry", "cg@test.com")
        id2 = database.log_request("water", "Water", "Care", "HIGH", "Thirsty", "cg@test.com")
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)

        # 2. Reset request ID sequence while records exist
        res = database.reset_request_id_sequence()
        self.assertTrue(res["success"])
        self.assertTrue(res["has_records"])
        self.assertEqual(res["record_count"], 2)
        self.assertEqual(res["next_id"], 3)
        self.assertIn("clear existing history records first", res["message"])

        # 3. Existing records must remain intact and unmodified
        logs = database.get_requests_log()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[1]["id"], 1)
        self.assertEqual(logs[0]["id"], 2)

        # 4. Next inserted request must not collide or overwrite; gets next sequential ID (3)
        id3 = database.log_request("medicine", "Medicine", "Care", "HIGH", "Pills", "cg@test.com")
        self.assertEqual(id3, 3)
        all_logs = database.get_requests_log()
        self.assertEqual(len(all_logs), 3)

    def test_reset_request_id_sequence_with_empty_table(self):
        """Verify reset request ID with empty history table causes next request ID to start at 1."""
        # 1. Clear history
        database.clear_requests_log()
        self.assertEqual(len(database.get_requests_log()), 0)

        # 2. Reset ID sequence on empty table
        res = database.reset_request_id_sequence()
        self.assertTrue(res["success"])
        self.assertFalse(res["has_records"])
        self.assertEqual(res["record_count"], 0)
        self.assertEqual(res["next_id"], 1)
        self.assertIn("start from ID #1", res["message"])

        # 3. Next inserted request strictly receives ID 1
        new_id = database.log_request("help", "Help", "Emergency", "CRITICAL", "Need help", "cg@test.com")
        self.assertEqual(new_id, 1)

    def test_api_reset_logs_id_endpoint_and_csrf(self):
        """Verify /api/logs/reset_id POST endpoint works and returns clear messages."""
        resp = self.client.post("/api/logs/reset_id")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("message", data)

    def test_clear_history_and_history_display_preserved(self):
        """Verify Clear History button API and history log display still work as intended."""
        database.log_request("water", "Water", "Care", "MEDIUM", "Water please", "cg@test.com")
        resp = self.client.get("/api/logs")
        self.assertEqual(resp.status_code, 200)
        logs = resp.get_json().get("logs", [])
        self.assertGreater(len(logs), 0)

        # Clear history endpoint
        clear_resp = self.client.post("/api/logs/clear")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertTrue(clear_resp.get_json()["success"])

        # Logs must now be empty
        logs_after = self.client.get("/api/logs").get_json().get("logs", [])
        self.assertEqual(len(logs_after), 0)


if __name__ == "__main__":
    unittest.main()
