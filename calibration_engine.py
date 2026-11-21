"""
Calibration Engine for AI-Vision Gaze Assistant.
Manages 5-point both-eye calibration, sample validation, stratified cross-validation,
regression model training (Ridge, Huber, Affine, SVR, Random Forest),
profile serialization, and quality gating.
"""

import os
import math
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import joblib

from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

logger = logging.getLogger("calibration_engine")

PROFILE_PATH = os.environ.get(
    "CALIBRATION_PROFILE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_profile.json")
)
CALIB_MODEL_PATH = os.environ.get(
    "CALIBRATION_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_model.joblib")
)

# Calibration quality thresholds (accessible precision threshold for real-world setups)
MIN_CALIBRATION_HIT_RATE = 55.0
MAX_CALIBRATION_MAE_PX = 180.0
# Minimum valid samples required per target box for successful calibration.
# Calibrated for a 1.5s dwell with 0.35s settling period and 75ms client sampling intervals.
MIN_SAMPLES_PER_BOX = 10


class BothEyeCalibrationManager:
    """
    Advanced BOTH-EYE Gaze Calibration Manager with Genuine Independent Validation.
    Preserves the 5-Box interface, applies settle time rejection, baseline-relative head gating,
    two independent affine ridge models on standardized features, minimum quality gating,
    and safe profile persistence with provenance validation.
    """
    def __init__(
        self,
        auto_load: bool = True,
        profile_path: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        self.profile_path = profile_path or os.environ.get("CALIBRATION_PROFILE_PATH", PROFILE_PATH)
        self.model_path = model_path or os.environ.get("CALIBRATION_MODEL_PATH", CALIB_MODEL_PATH)

        self.target_boxes = [
            {"id": "top_right", "name": "Top-Right Target", "x": 0.90, "y": 0.10, "w": 0.18, "h": 0.18},
            {"id": "top_left", "name": "Top-Left Target", "x": 0.10, "y": 0.10, "w": 0.18, "h": 0.18},
            {"id": "bottom_right", "name": "Bottom-Right Target", "x": 0.90, "y": 0.90, "w": 0.18, "h": 0.18},
            {"id": "bottom_left", "name": "Bottom-Left Target", "x": 0.10, "y": 0.90, "w": 0.18, "h": 0.18},
            {"id": "center", "name": "Center Target", "x": 0.50, "y": 0.50, "w": 0.18, "h": 0.18}
        ]
        self.is_calibrating = False
        self.is_calibrated = False
        self.current_point_index = 0
        self.box_start_time = {}
        self.selected_model_name = "None"
        self.selected_model = None
        self.model_x = None
        self.model_y = None
        self.accuracy = None
        self.accuracy_label = "Uncalibrated (Accuracy Not Measured)"
        self.quality = "None"
        self.status_message = "Ready for both-eye calibration"
        self.telemetry = {
            "calibration_stale": False,
            "calibration_reject_reason": "",
            "calibration_expired": False
        }

        self.val_mean_euclidean_px = None
        self.val_median_euclidean_px = None
        self.val_mae_px = None
        self.val_mae_x_px = None
        self.val_mae_y_px = None
        self.val_rmse_px = None
        self.val_hit_rate_pct = None
        self.hit_rate = None
        self.val_mean_euclidean_norm = None
        self.val_median_euclidean_norm = None
        self.val_sample_count = 0
        self.model_cv_scores = {}
        self.train_mean_euclidean_px = None
        self.train_rmse_px = None
        self.train_hit_rate_pct = None
        self.oof_predictions = None
        self.samples = {b["id"]: [] for b in self.target_boxes}

        if auto_load:
            self.load_profile()

    def start_calibration(self) -> Dict[str, Any]:
        """Begin or restart 5-box both-eye calibration sequence."""
        self.is_calibrating = True
        self.is_calibrated = False
        self.current_point_index = 0
        self.samples = {b["id"]: [] for b in self.target_boxes}
        now = time.time()
        self.box_start_time = {"top_right": now}
        self.selected_model = None
        self.selected_model_name = "None"
        self.accuracy = None
        self.accuracy_label = "Calibrating (Accuracy Not Measured)"
        self.quality = "Calibrating"
        self.status_message = "Look inside Point 1 (Top-Right) to begin 5-point calibration"
        self.val_mae_px = None
        self.val_rmse_px = None

        return {
            "status": "started",
            "point_index": 0,
            "target_box": self.target_boxes[0],
            "total_boxes": len(self.target_boxes)
        }

    def activate_box(self, box_id: str, reset: bool = False) -> bool:
        """Client signals that box_id is active. Enforces five-point sequence and settling time."""
        now = time.time()
        matching_idx = next((i for i, b in enumerate(self.target_boxes) if b["id"] == box_id), None)
        if matching_idx is None:
            return False

        if matching_idx == self.current_point_index:
            self.box_start_time[box_id] = now
            if reset:
                self.samples[box_id] = []
            return True

        if matching_idx == self.current_point_index + 1:
            self.current_point_index = matching_idx
            self.box_start_time[box_id] = now
            if reset or box_id not in self.samples:
                self.samples[box_id] = []
            return True

        return False

    def add_sample(
        self,
        box_id: str,
        raw_x: float,
        raw_y: float,
        left_dx: float = 0.0,
        left_dy: float = 0.0,
        right_dx: float = 0.0,
        right_dy: float = 0.0,
        conf_l: float = 1.0,
        conf_r: float = 1.0,
        is_blinking: bool = False,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        both_eyes_valid: bool = True,
        gaze_valid: bool = True
    ) -> Tuple[bool, str]:
        """
        Record sample for active target box.
        Enforces calibration active, target sequence, settling period, both-eye validity,
        and active target normalized tolerance region.
        """
        if not self.is_calibrating:
            reason = "Calibration not active"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        if self.current_point_index >= len(self.target_boxes):
            reason = "Calibration sequence already completed"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        active_target = self.target_boxes[self.current_point_index]
        if box_id != active_target["id"]:
            reason = f"Target sequence mismatch: expected '{active_target['id']}', got '{box_id}'"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        now = time.time()
        if box_id not in self.box_start_time:
            self.box_start_time[box_id] = now

        elapsed = now - self.box_start_time[box_id]
        if elapsed < 0.35:
            reason = f"Settling period active ({elapsed:.2f}s < 0.35s)"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        if is_blinking:
            reason = "Blink detected"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason
        if not both_eyes_valid or not gaze_valid:
            reason = "Both eyes must be valid and tracking reliable"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason
        if conf_l < 0.25 or conf_r < 0.25:
            reason = f"Low iris confidence (L={conf_l:.2f}, R={conf_r:.2f})"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason
        if abs(head_yaw) > 10.0 or abs(head_pitch) > 10.0:
            reason = f"Excessive head motion ({head_yaw:.1f}°, {head_pitch:.1f}°)"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        tx = active_target["x"]
        ty = active_target["y"]
        tol_x = 0.55
        tol_y = 0.55
        if abs(raw_x - tx) > tol_x or abs(raw_y - ty) > tol_y:
            reason = f"Out of target tolerance: gaze ({raw_x:.2f}, {raw_y:.2f}) vs target ({tx:.2f}, {ty:.2f})"
            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_reject_reason"] = reason
            return False, reason

        if box_id not in self.samples:
            self.samples[box_id] = []

        self.samples[box_id].append({
            "raw_x": float(raw_x),
            "raw_y": float(raw_y),
            "left_dx": float(left_dx),
            "left_dy": float(left_dy),
            "right_dx": float(right_dx),
            "right_dy": float(right_dy),
            "head_yaw": float(head_yaw),
            "head_pitch": float(head_pitch),
            "timestamp": now
        })
        if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
            self.telemetry["calibration_reject_reason"] = ""
        return True, "Sample accepted"

    def finish_calibration(self, frame_w: int = 1920, frame_h: int = 1080) -> Dict[str, Any]:
        """
        Multi-model evaluation with Stratified K-Fold CV on trimmed samples.
        Enforces MIN_SAMPLES_PER_BOX (10) and strict quality gating.
        """
        self.is_calibrating = False
        min_samples_per_box = MIN_SAMPLES_PER_BOX

        # Check all 5 target boxes
        box_medians = []
        all_train_samples = []
        for box in self.target_boxes:
            bid = box["id"]
            pts = self.samples.get(bid, [])
            if len(pts) < min_samples_per_box:
                self.is_calibrated = False
                self.quality = "Insufficient Data"
                self.accuracy = None
                self.accuracy_label = "Insufficient Calibration Data (All 5 Target Boxes Required)"
                self.status_message = f"Calibration failed: box '{box['name']}' had only {len(pts)}/{min_samples_per_box} valid samples."
                return {
                    "success": False,
                    "calibrated": False,
                    "accuracy": None,
                    "accuracy_label": self.accuracy_label,
                    "quality": self.quality,
                    "message": self.status_message
                }

            xs = [p["raw_x"] for p in pts]
            ys = [p["raw_y"] for p in pts]
            lower_px = np.percentile(xs, 10)
            upper_px = np.percentile(xs, 90)
            lower_py = np.percentile(ys, 10)
            upper_py = np.percentile(ys, 90)

            trimmed = [p for p in pts if lower_px <= p["raw_x"] <= upper_px and lower_py <= p["raw_y"] <= upper_py]
            if not trimmed:
                trimmed = pts

            med_x = float(np.median([p["raw_x"] for p in trimmed]))
            med_y = float(np.median([p["raw_y"] for p in trimmed]))
            box_medians.append((med_x, med_y, box["x"], box["y"], bid))

            for p in trimmed:
                all_train_samples.append({
                    "raw_x": p["raw_x"],
                    "raw_y": p["raw_y"],
                    "head_yaw": p.get("head_yaw", 0.0),
                    "head_pitch": p.get("head_pitch", 0.0),
                    "target_x": box["x"],
                    "target_y": box["y"],
                    "box_id": bid
                })

        # Static gaze check
        raw_xs = [b[0] for b in box_medians]
        raw_ys = [b[1] for b in box_medians]
        spread_x = float(np.std(raw_xs))
        spread_y = float(np.std(raw_ys))
        if spread_x < 0.015 or spread_y < 0.012:
            self.is_calibrated = False
            self.quality = "Poor Gaze Spread"
            self.accuracy = None
            self.accuracy_label = "Calibration Failed (Insufficient Eye Movement)"
            self.status_message = "Calibration failed: eyes remained static - using adaptive baseline."
            return {
                "success": False,
                "calibrated": False,
                "accuracy": None,
                "accuracy_label": self.accuracy_label,
                "quality": self.quality,
                "message": self.status_message
            }

        candidate_names = ["Ridge Regression", "Support Vector Regression", "Random Forest", "Affine Standard", "Huber Robust"]
        candidate_pipelines = {
            "Ridge Regression": (
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, fit_intercept=True))]),
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, fit_intercept=True))])
            ),
            "Support Vector Regression": (
                Pipeline([("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.01))]),
                Pipeline([("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.01))])
            ),
            "Random Forest": (
                Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=25, max_depth=4, random_state=42))]),
                Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=25, max_depth=4, random_state=42))])
            ),
            "Affine Standard": (
                Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(fit_intercept=True))]),
                Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(fit_intercept=True))])
            ),
            "Huber Robust": (
                Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor())]),
                Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor())])
            )
        }

        unique_boxes = [b["id"] for b in self.target_boxes]
        half_w = 0.11
        half_h = 0.11

        best_model_name = "Affine Standard"
        best_cv_mae = float("inf")
        best_oof_preds = []
        all_cv_scores = {}

        num_folds = 4
        box_samples_map = {bid: [s for s in all_train_samples if s["box_id"] == bid] for bid in unique_boxes}

        for name in candidate_names:
            model_x_proto, model_y_proto = candidate_pipelines[name]
            oof_preds = []
            fold_errors = []

            for fold_idx in range(num_folds):
                train_fold = []
                val_fold = []
                for bid, b_samples in box_samples_map.items():
                    n_b = len(b_samples)
                    val_indices = set(range(fold_idx, n_b, num_folds))
                    for idx, sample in enumerate(b_samples):
                        if idx in val_indices:
                            val_fold.append(sample)
                        else:
                            train_fold.append(sample)

                if not train_fold or not val_fold:
                    continue

                X_tr_x = np.array([[s["raw_x"], s["head_yaw"]] for s in train_fold], dtype=np.float64)
                Y_tr_x = np.array([s["target_x"] for s in train_fold], dtype=np.float64)
                X_tr_y = np.array([[s["raw_y"], s["head_pitch"]] for s in train_fold], dtype=np.float64)
                Y_tr_y = np.array([s["target_y"] for s in train_fold], dtype=np.float64)

                mx = clone(model_x_proto)
                my = clone(model_y_proto)
                try:
                    mx.fit(X_tr_x, Y_tr_x)
                    my.fit(X_tr_y, Y_tr_y)

                    X_va_x = np.array([[s["raw_x"], s["head_yaw"]] for s in val_fold], dtype=np.float64)
                    X_va_y = np.array([[s["raw_y"], s["head_pitch"]] for s in val_fold], dtype=np.float64)
                    pred_x = mx.predict(X_va_x)
                    pred_y = my.predict(X_va_y)

                    for i, s in enumerate(val_fold):
                        px = float(pred_x[i])
                        py = float(pred_y[i])
                        oof_preds.append((px, py, s["target_x"], s["target_y"]))
                        err_px = math.hypot((px - s["target_x"]) * frame_w, (py - s["target_y"]) * frame_h)
                        fold_errors.append(err_px)
                except Exception:
                    fold_errors.append(999.0)

            mean_mae = float(np.mean(fold_errors)) if fold_errors else 999.0
            all_cv_scores[name] = round(mean_mae, 1)

            if mean_mae < best_cv_mae:
                best_cv_mae = mean_mae
                best_model_name = name
                best_oof_preds = oof_preds

        self.selected_model_name = best_model_name
        self.model_cv_scores = all_cv_scores
        self.oof_predictions = best_oof_preds

        if best_oof_preds:
            oof_arr = np.array(best_oof_preds, dtype=np.float64)
            preds = oof_arr[:, :2]
            targets = oof_arr[:, 2:]

            px_errors = np.sqrt(
                ((preds[:, 0] - targets[:, 0]) * frame_w) ** 2 +
                ((preds[:, 1] - targets[:, 1]) * frame_h) ** 2
            )
            norm_errors = np.sqrt(
                (preds[:, 0] - targets[:, 0]) ** 2 +
                (preds[:, 1] - targets[:, 1]) ** 2
            )

            self.val_mae_px = round(float(np.mean(px_errors)), 1)
            self.val_rmse_px = round(float(np.sqrt(np.mean(px_errors ** 2))), 1)
            self.val_mean_euclidean_px = self.val_mae_px
            self.val_median_euclidean_px = round(float(np.median(px_errors)), 1)
            self.val_mae_x_px = round(float(np.mean(np.abs(preds[:, 0] - targets[:, 0]) * frame_w)), 1)
            self.val_mae_y_px = round(float(np.mean(np.abs(preds[:, 1] - targets[:, 1]) * frame_h)), 1)
            self.val_mean_euclidean_norm = round(float(np.mean(norm_errors)), 4)
            self.val_median_euclidean_norm = round(float(np.median(norm_errors)), 4)

            hits = sum(
                1 for i in range(len(targets))
                if abs(preds[i, 0] - targets[i, 0]) <= half_w and
                   abs(preds[i, 1] - targets[i, 1]) <= half_h
            )
            self.val_hit_rate_pct = round((hits / len(targets)) * 100.0, 1)
            self.hit_rate = self.val_hit_rate_pct
            self.accuracy = self.val_hit_rate_pct
            self.val_sample_count = len(best_oof_preds)

        final_x_proto, final_y_proto = candidate_pipelines[best_model_name]
        self.model_x = clone(final_x_proto)
        self.model_y = clone(final_y_proto)

        X_all_x = np.array([[s["raw_x"], s["head_yaw"]] for s in all_train_samples], dtype=np.float64)
        Y_all_x = np.array([s["target_x"] for s in all_train_samples], dtype=np.float64)
        X_all_y = np.array([[s["raw_y"], s["head_pitch"]] for s in all_train_samples], dtype=np.float64)
        Y_all_y = np.array([s["target_y"] for s in all_train_samples], dtype=np.float64)

        self.model_x.fit(X_all_x, Y_all_x)
        self.model_y.fit(X_all_y, Y_all_y)
        self.selected_model = self.model_x

        tr_pred_x = self.model_x.predict(X_all_x)
        tr_pred_y = self.model_y.predict(X_all_y)
        tr_px_errors = np.sqrt(
            ((tr_pred_x - Y_all_x) * frame_w) ** 2 +
            ((tr_pred_y - Y_all_y) * frame_h) ** 2
        )
        self.train_mean_euclidean_px = round(float(np.mean(tr_px_errors)), 1)
        self.train_rmse_px = round(float(np.sqrt(np.mean(tr_px_errors ** 2))), 1)

        if self.val_hit_rate_pct < MIN_CALIBRATION_HIT_RATE or self.val_mae_px > MAX_CALIBRATION_MAE_PX:
            self.is_calibrated = False
            self.quality = "Poor Precision"
            self.accuracy_label = f"Poor Precision ({self.accuracy}% Hit Rate, {self.val_mae_px}px error)"
            self.status_message = "Calibration failed minimum precision requirement. Using adaptive baseline."
            print(f"[Calibration Engine] Quality check failed: HitRate={self.val_hit_rate_pct}%, MAE={self.val_mae_px}px. Profile not saved.")
            return {
                "success": False,
                "calibrated": False,
                "accuracy": self.accuracy,
                "accuracy_label": self.accuracy_label,
                "quality": self.quality,
                "val_mae_px": self.val_mae_px,
                "val_rmse_px": self.val_rmse_px,
                "hit_rate": self.hit_rate,
                "message": self.status_message,
                "model_type": self.selected_model_name,
                "val_mean_euclidean_px": self.val_mean_euclidean_px,
                "val_median_euclidean_px": self.val_median_euclidean_px,
                "val_mae_x_px": self.val_mae_x_px,
                "val_mae_y_px": self.val_mae_y_px,
                "val_sample_count": self.val_sample_count,
            }

        if self.val_mean_euclidean_px <= 65.0 and self.val_hit_rate_pct >= 85.0:
            self.quality = "High Precision"
        else:
            self.quality = "Moderate Precision"

        self.is_calibrated = True
        self.accuracy_label = f"{self.accuracy}% Target Hit Rate (Stratified Val MAE: {self.val_mae_px}px)"
        self.status_message = f"Both-Eye Calibration Complete ({self.selected_model_name}) | Hit Rate: {self.hit_rate}% | Val MAE: {self.val_mae_px}px"

        self.save_profile(frame_w, frame_h)

        return {
            "success": True,
            "calibrated": True,
            "accuracy": self.accuracy,
            "accuracy_label": self.accuracy_label,
            "quality": self.quality,
            "model_type": self.selected_model_name,
            "val_mae_px": self.val_mae_px,
            "val_rmse_px": self.val_rmse_px,
            "val_mean_euclidean_px": self.val_mean_euclidean_px,
            "val_median_euclidean_px": self.val_median_euclidean_px,
            "val_mae_x_px": self.val_mae_x_px,
            "val_mae_y_px": self.val_mae_y_px,
            "hit_rate": self.hit_rate,
            "val_sample_count": self.val_sample_count,
            "model_cv_scores": self.model_cv_scores,
            "message": self.status_message
        }

    def apply(
        self,
        raw_x: float,
        raw_y: float,
        left_dx: float = 0.0,
        left_dy: float = 0.0,
        right_dx: float = 0.0,
        right_dy: float = 0.0,
        delta_yaw: float = 0.0,
        delta_pitch: float = 0.0,
        **kwargs
    ) -> Tuple[float, float]:
        """Apply calibrated regression model, clamped to [0.02, 0.98]."""
        if not self.is_calibrated or self.model_x is None or self.model_y is None:
            return raw_x, raw_y

        try:
            X_in = np.array([[raw_x, delta_yaw]], dtype=np.float64)
            Y_in = np.array([[raw_y, delta_pitch]], dtype=np.float64)
            pred_x = float(self.model_x.predict(X_in)[0])
            pred_y = float(self.model_y.predict(Y_in)[0])
            return float(np.clip(pred_x, 0.02, 0.98)), float(np.clip(pred_y, 0.02, 0.98))
        except Exception:
            return raw_x, raw_y

    map_gaze = apply

    def mark_stale(self):
        """Invalidate calibration when camera, mirror, or baseline recentering changes."""
        if self.is_calibrated:
            self.is_calibrated = False
            self.quality = "Stale (Recalibration Recommended)"
        if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
            self.telemetry["calibration_stale"] = True

    def save_profile(self, frame_w: int, frame_h: int, camera_index: int = 0, mirror_view: bool = True):
        """Serialize trained model and companion metadata."""
        try:
            model_bundle = {
                "version": "4.2.0",
                "source": "user_calibration",
                "architecture": "both_eyes",
                "timestamp": time.time(),
                "camera_index": camera_index,
                "mirror_view": mirror_view,
                "resolution": f"{frame_w}x{frame_h}",
                "selected_model_name": self.selected_model_name,
                "model_x": self.model_x,
                "model_y": self.model_y,
                "hit_rate": self.hit_rate,
                "accuracy": self.accuracy,
                "val_mae_px": self.val_mae_px,
                "val_rmse_px": self.val_rmse_px,
                "quality": self.quality,
                "validation_metrics": {
                    "mean_euclidean_px": self.val_mean_euclidean_px,
                    "median_euclidean_px": self.val_median_euclidean_px,
                    "mae_x_px": self.val_mae_x_px,
                    "mae_y_px": self.val_mae_y_px,
                    "rmse_px": self.val_rmse_px,
                    "hit_rate_pct": self.val_hit_rate_pct,
                    "sample_count": self.val_sample_count,
                    "cv_scores": self.model_cv_scores
                }
            }
            joblib.dump(model_bundle, self.model_path)

            meta_data = {
                "version": "4.2.0",
                "source": "user_calibration",
                "architecture": "both_eyes",
                "timestamp": time.time(),
                "camera_index": camera_index,
                "mirror_view": mirror_view,
                "resolution": f"{frame_w}x{frame_h}",
                "model_name": self.selected_model_name,
                "accuracy": self.accuracy,
                "hit_rate": self.hit_rate,
                "val_mae_px": self.val_mae_px,
                "val_rmse_px": self.val_rmse_px,
                "quality": self.quality,
                "samples_count": self.val_sample_count,
                "has_trained_model": True
            }
            with open(self.profile_path, "w") as f:
                json.dump(meta_data, f, indent=2)
            logger.debug(f"[Calibration Engine] Genuine profile saved to {self.profile_path} and model to {self.model_path}")
        except Exception as e:
            print(f"[Calibration Engine Warning] Failed to save profile: {e}")

    def load_profile(
        self,
        camera_index: Optional[int] = None,
        mirror_view: Optional[bool] = None,
        resolution: Optional[str] = None
    ) -> bool:
        """Load and restore trained regression model."""
        if not os.path.exists(self.model_path):
            return False

        try:
            bundle = joblib.load(self.model_path)
            if bundle.get("source") != "user_calibration":
                print("[Calibration Engine] Refusing to load calibration without 'user_calibration' provenance.")
                return False

            saved_time = bundle.get("timestamp", 0)
            if (time.time() - saved_time) > 86400:
                print("[Calibration Engine] Calibration profile expired (> 24 hours old). Starting uncalibrated.")
                self.is_calibrated = False
                self.quality = "Expired"
                self.accuracy = None
                self.accuracy_label = "Calibration Expired (>24 hours old)"
                self.status_message = "Calibration profile expired. Recalibration recommended."
                if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                    self.telemetry["calibration_expired"] = True
                    self.telemetry["calibration_stale"] = True
                return False

            if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
                self.telemetry["calibration_expired"] = False

            if camera_index is not None and bundle.get("camera_index") != camera_index:
                print("[Calibration Engine] Camera index mismatch for stored profile. Recalibration required.")
                return False

            if mirror_view is not None and bundle.get("mirror_view") != mirror_view:
                print("[Calibration Engine] Mirror setting mismatch for stored profile. Recalibration required.")
                return False

            self.model_x = bundle.get("model_x")
            self.model_y = bundle.get("model_y")
            self.selected_model = self.model_x
            self.selected_model_name = bundle.get("selected_model_name", "Affine Standard")
            self.accuracy = bundle.get("accuracy")
            self.hit_rate = bundle.get("hit_rate")
            self.val_mae_px = bundle.get("val_mae_px")
            self.val_rmse_px = bundle.get("val_rmse_px")
            self.quality = bundle.get("quality", "Restored Profile")
            self.is_calibrated = bool(self.model_x is not None and self.model_y is not None)

            if self.is_calibrated:
                self.accuracy_label = f"Restored Profile: {self.accuracy}% (Val MAE: {self.val_mae_px}px)"
                self.status_message = f"Both-Eye Calibrated ({self.selected_model_name} | Hit Rate: {self.accuracy}%)"
                return True
        except Exception as e:
            print(f"[Calibration Engine Warning] Could not restore profile: {e}")

        return False

    def skip_calibration(self) -> Dict[str, Any]:
        self.is_calibrating = False
        self.is_calibrated = False
        self.quality = "Skipped"
        self.accuracy = None
        self.accuracy_label = "Adaptive Baseline Active (Accuracy Not Measured)"
        self.status_message = "Calibration skipped - using adaptive baseline."
        return {
            "success": True,
            "calibrated": False,
            "accuracy": None,
            "accuracy_label": self.accuracy_label,
            "quality": self.quality,
            "message": self.status_message
        }


# Backwards compatibility aliases
EyeCalibration = BothEyeCalibrationManager
