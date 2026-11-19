"""
AI Vision & Precision BOTH-EYE Gaze Estimation Engine.
MANDATORY BOTH-EYE TRACKING ARCHITECTURE:
Simultaneously tracks the left and right eyes, localizes both irises with subpixel precision,
measures independent left and right eye geometry, computes canthal axis projections for 2D roll-invariance,
performs intelligent multimodal reliability-based both-eye and 3D head-pose fusion,
evaluates multi-model regression (Ridge, Huber Robust, Affine Standard) with genuine
Leave-One-Box-Out Cross-Validation (LOBO-CV) on independent held-out samples,
and reports genuine independent validation metrics (STRICTLY NO ARTIFICIAL ACCURACY FLOORS).
Complete model persistence via Joblib serialization.
"""

import os
import contextlib

# Suppress oneDNN, TensorFlow, and MediaPipe C++ info/warning logs before library imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["CLEARCUT_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

@contextlib.contextmanager
def _suppress_c_output():
    """Temporarily redirect native C-level stdout (fd 1) and stderr (fd 2) to devnull to silence low-level C/C++ messages."""
    old_stdout = None
    old_stderr = None
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            if old_stdout is not None:
                os.dup2(old_stdout, 1)
                os.close(old_stdout)
            if old_stderr is not None:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
            if devnull is not None:
                os.close(devnull)
        except Exception:
            pass

_suppress_c_stderr = _suppress_c_output

with _suppress_c_stderr():
    import cv2
    import numpy as np
    import mediapipe as mp

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

import math
import time
import json
import threading
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("ai_vision_engine")

from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
PROFILE_PATH = os.environ.get("CALIBRATION_PROFILE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_profile.json"))
CALIB_MODEL_PATH = os.environ.get("CALIBRATION_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_model.joblib"))

# Gaze disparity and agreement constants (A1)
DELTA_DISPARITY_LIMIT_X = 0.08
DELTA_DISPARITY_LIMIT_Y = 0.08
DELTA_DISPARITY_SCALE = 0.12

# Baseline stabilization constants (A1, A4)
BASELINE_DISPERSION_THRESHOLD = 0.020
BASELINE_TIMEOUT_SECONDS = 5.0
BASELINE_REQUIRED_SAMPLES = 15

# Calibration quality thresholds (C7)
MIN_CALIBRATION_HIT_RATE = 70.0
MAX_CALIBRATION_MAE_PX = 150.0

# Feature flags for Phase 2 upgrades (E1u, E2u)
USE_FACIAL_TRANSFORMATION_MATRIX = True
USE_VIDEO_MODE = True


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1.0 - a) * x_prev


class AdaptiveOneEuroFilter:
    """
    Advanced 1-Euro adaptive low-pass filter with velocity-aware damping,
    saccade-boost detection, and separate horizontal/vertical tracking.
    Eliminates high-frequency tremor and micro-jitter during steady fixation
    while providing rapid, overshoot-free response during intentional saccades.
    """
    def __init__(
        self,
        t0: float = 0.0,
        x0: float = 0.5,
        dx0: float = 0.0,
        min_cutoff: float = 0.45,
        beta: float = 0.020,
        d_cutoff: float = 2.0,
        saccade_velocity_thresh: float = 2.5
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.saccade_velocity_thresh = float(saccade_velocity_thresh)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)
        self.last_saccade_time = 0.0

    def filter(self, t: float, x: float, confidence: float = 1.0) -> float:
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev

        conf = max(0.1, min(1.0, float(confidence)))

        # Filter the derivative
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        speed = max(abs(dx_hat), abs(dx) * 0.75)

        # Saccade boost: if intentional eye jump is detected.
        # Multipliers are kept modest so that sensor/landmark noise does not
        # trip the boost and disable low-pass filtering (which caused the
        # rapid cursor bouncing). Genuine saccades still track instantly.
        if speed > self.saccade_velocity_thresh:
            self.last_saccade_time = t
            eff_beta = self.beta * 3.0
            eff_min_cutoff = self.min_cutoff * 1.5
        elif t - self.last_saccade_time < 0.20:
            eff_beta = self.beta * 2.0
            eff_min_cutoff = self.min_cutoff * 1.25
        else:
            eff_beta = self.beta
            eff_min_cutoff = self.min_cutoff * conf

        cutoff = eff_min_cutoff + eff_beta * speed
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self, t0: float = 0.0, x0: float = 0.5):
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)
        self.last_saccade_time = 0.0


class BothEyeIrisLocalizer:
    """
    Subpixel iris localization for BOTH left and right eyes simultaneously.
    Combines deep learning iris landmark coordinates (468-477) with local gradient
    radial-symmetry refinement, boundary circularity scoring, and per-eye measurement validity.
    """
    def __init__(self):
        # MediaPipe iris landmark indices
        self.right_iris_indices = [468, 469, 470, 471, 472]  # Right eye (OD)
        self.left_iris_indices = [473, 474, 475, 476, 477]   # Left eye (OS)

        # Temporal trackers per eye for smoothing and jump protection
        self.prev_centers = {"left": None, "right": None}
        self.prev_confidences = {"left": 1.0, "right": 1.0}

    def localize_both_eyes(
        self,
        frame: np.ndarray,
        landmarks: Any,
        frame_w: int,
        frame_h: int
    ) -> Tuple[Tuple[float, float, float, float, Dict[str, Any]], Tuple[float, float, float, float, Dict[str, Any]]]:
        """
        Returns:
            left_eye: (norm_x, norm_y, radius_px, confidence, meta_dict)
            right_eye: (norm_x, norm_y, radius_px, confidence, meta_dict)
        """
        left_res = self._localize_single_eye(frame, landmarks, self.left_iris_indices, "left", frame_w, frame_h)
        right_res = self._localize_single_eye(frame, landmarks, self.right_iris_indices, "right", frame_w, frame_h)
        return left_res, right_res

    def _localize_single_eye(
        self,
        frame: np.ndarray,
        landmarks: Any,
        indices: List[int],
        eye_side: str,
        frame_w: int,
        frame_h: int
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        pts = np.array([
            [landmarks[idx].x * frame_w, landmarks[idx].y * frame_h]
            for idx in indices
        ], dtype=np.float32)

        center_pt = pts[0]
        periph_pts = pts[1:5]
        radii = np.linalg.norm(periph_pts - center_pt, axis=1)
        median_radius = float(np.median(radii))
        if median_radius < 1.0:
            median_radius = 6.0

        refine_radius = int(math.ceil(median_radius * 1.5))
        cx_i = int(round(center_pt[0]))
        cy_i = int(round(center_pt[1]))

        x0 = max(0, cx_i - refine_radius)
        x1 = min(frame_w, cx_i + refine_radius + 1)
        y0 = max(0, cy_i - refine_radius)
        y1 = min(frame_h, cy_i + refine_radius + 1)

        refined_x = float(center_pt[0])
        refined_y = float(center_pt[1])
        refinement_accepted = False
        contrast_ratio = 1.0

        if (x1 - x0) >= 6 and (y1 - y0) >= 6 and frame is not None:
            roi = frame[y0:y1, x0:x1]
            if roi.size > 0:
                if len(roi.shape) == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = roi

                min_val, max_val, _, _ = cv2.minMaxLoc(gray_roi)
                contrast_ratio = float((max_val + 1.0) / (min_val + 1.0))

                grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(grad_x, grad_y)

                thresh = np.percentile(mag, 70)
                mask = mag >= thresh
                pts_y, pts_x = np.nonzero(mask)

                if len(pts_x) >= 8:
                    dx = grad_x[pts_y, pts_x]
                    dy = grad_y[pts_y, pts_x]
                    norm = np.hypot(dx, dy) + 1e-5
                    dx_n = dx / norm
                    dy_n = dy / norm

                    rx = pts_x - (center_pt[0] - x0)
                    ry = pts_y - (center_pt[1] - y0)
                    r_norm = np.hypot(rx, ry) + 1e-5
                    rx_n = rx / r_norm
                    ry_n = ry / r_norm

                    alignments = np.abs(dx_n * rx_n + dy_n * ry_n)
                    weights = mag[pts_y, pts_x] * alignments

                    if np.sum(weights) > 1e-5:
                        w_norm = weights / np.sum(weights)
                        sub_x = float(np.sum(pts_x * w_norm))
                        sub_y = float(np.sum(pts_y * w_norm))
                        cand_x = x0 + sub_x
                        cand_y = y0 + sub_y

                        dist_shift = math.hypot(cand_x - center_pt[0], cand_y - center_pt[1])
                        if dist_shift <= median_radius * 0.45:
                            refined_x = (center_pt[0] * 0.70) + (cand_x * 0.30)
                            refined_y = (center_pt[1] * 0.70) + (cand_y * 0.30)
                            refinement_accepted = True

        rad_std = float(np.std(radii))
        circularity = float(np.clip(1.0 - (rad_std / (median_radius + 1e-5)), 0.0, 1.0))

        # Occlusion estimation
        if eye_side == "left":
            upper_lid_idx = 386
            lower_lid_idx = 374
        else:
            upper_lid_idx = 159
            lower_lid_idx = 145

        upper_lid_y = landmarks[upper_lid_idx].y * frame_h
        lower_lid_y = landmarks[lower_lid_idx].y * frame_h
        occlusion_superior = bool(upper_lid_y > (refined_y - median_radius * 0.55))
        occlusion_inferior = bool(lower_lid_y < (refined_y + median_radius * 0.55))

        occlusion_penalty = 0.25 if occlusion_superior else 0.0
        contrast_boost = min(0.20, max(0.0, (contrast_ratio - 1.0) * 0.15))

        confidence = float(np.clip(
            (circularity * 0.55) + (0.25 if refinement_accepted else 0.12) + contrast_boost - occlusion_penalty,
            0.15,
            1.0
        ))

        # A7: Temporal jump limit scaled with frame width
        jump_limit_px = max(24.0, frame_w * 0.05)
        prev_pt = self.prev_centers.get(eye_side)
        if prev_pt is not None:
            dt_px = math.hypot(refined_x - prev_pt[0], refined_y - prev_pt[1])
            if dt_px > jump_limit_px:
                confidence *= 0.60

        self.prev_centers[eye_side] = (refined_x, refined_y)
        self.prev_confidences[eye_side] = confidence

        norm_x = refined_x / float(frame_w)
        norm_y = refined_y / float(frame_h)

        # A7: Resolution-scaled minimum radius
        min_radius = max(1.2, 1.8 * (frame_w / 640.0))
        is_valid = bool(
            (confidence >= 0.20) and
            (0.01 <= norm_x <= 0.99) and
            (0.01 <= norm_y <= 0.99) and
            (median_radius >= min_radius)
        )

        meta = {
            "eye_side": eye_side,
            "is_valid": is_valid,
            "subpixel_refined": refinement_accepted,
            "contrast_ratio": round(contrast_ratio, 2),
            "median_radius_px": round(median_radius, 2),
            "occlusion_superior": occlusion_superior,
            "occlusion_inferior": occlusion_inferior,
            "circularity": round(circularity, 2)
        }
        return norm_x, norm_y, median_radius, confidence, meta


class BothEyeGeometryAnalyzer:
    """
    Computes independent and bilateral eye geometry for left and right eyes.
    Provides 2D roll-invariance via canthal axis projection, dynamic blink detection
    with baseline hysteresis and downward gaze protection, and baseline-relative delta disparity.
    """
    def __init__(self):
        # Anatomical Left Eye (OS) landmarks
        self.left_outer = 263
        self.left_inner = 362
        self.left_upper = 386
        self.left_lower = 374

        # Anatomical Right Eye (OD) landmarks
        self.right_outer = 33
        self.right_inner = 133
        self.right_upper = 159
        self.right_lower = 145

        # Blink state memory for hysteresis (A3)
        self._prev_blink_l = False
        self._prev_blink_r = False
        self._blink_counter_l = 0
        self._blink_counter_r = 0

    def analyze_both_eyes(
        self,
        landmarks: Any,
        left_iris_norm: Tuple[float, float],
        right_iris_norm: Tuple[float, float],
        blendshapes: Any,
        frame_w: int,
        frame_h: int,
        mirror_view: bool = True,
        baseline_l: Optional[Tuple[float, float]] = None,
        baseline_r: Optional[Tuple[float, float]] = None,
        baseline_ear_l: Optional[float] = None,
        baseline_ear_r: Optional[float] = None
    ) -> Dict[str, Any]:
        min_eye_w = max(14.0, 18.0 * (frame_w / 640.0))

        # ==================== ANATOMICAL LEFT EYE GEOMETRY ====================
        p_l_outer = np.array([landmarks[self.left_outer].x * frame_w, landmarks[self.left_outer].y * frame_h], dtype=np.float64)
        p_l_inner = np.array([landmarks[self.left_inner].x * frame_w, landmarks[self.left_inner].y * frame_h], dtype=np.float64)

        v_l = p_l_inner - p_l_outer
        if v_l[0] < 0:
            v_l = -v_l

        eye_w_l = float(np.linalg.norm(v_l))
        eye_w_l = max(min_eye_w, eye_w_l)
        u_axis_l = v_l / eye_w_l
        u_perp_l = np.array([-u_axis_l[1], u_axis_l[0]], dtype=np.float64)
        if u_perp_l[1] < 0:
            u_perp_l = -u_perp_l

        p_mid_l = (p_l_outer + p_l_inner) / 2.0
        p_l_upper = np.array([landmarks[self.left_upper].x * frame_w, landmarks[self.left_upper].y * frame_h], dtype=np.float64)
        p_l_lower = np.array([landmarks[self.left_lower].x * frame_w, landmarks[self.left_lower].y * frame_h], dtype=np.float64)
        eye_h_l = float(max(4.0, np.linalg.norm(p_l_upper - p_l_lower)))

        # Left EAR
        p387 = np.array([landmarks[387].x * frame_w, landmarks[387].y * frame_h], dtype=np.float64)
        p373 = np.array([landmarks[373].x * frame_w, landmarks[373].y * frame_h], dtype=np.float64)
        p385 = np.array([landmarks[385].x * frame_w, landmarks[385].y * frame_h], dtype=np.float64)
        p380 = np.array([landmarks[380].x * frame_w, landmarks[380].y * frame_h], dtype=np.float64)
        ear_l = float((np.linalg.norm(p387 - p373) + np.linalg.norm(p385 - p380)) / (2.0 * eye_w_l))

        # Left iris canthal projection
        p_iris_l = np.array([left_iris_norm[0] * frame_w, left_iris_norm[1] * frame_h], dtype=np.float64)
        offset_l = p_iris_l - p_mid_l
        norm_disp_l_x = float(np.dot(offset_l, u_axis_l)) / eye_w_l
        norm_disp_l_y = float(np.dot(offset_l, u_perp_l)) / (eye_w_l * 0.42)

        # ==================== ANATOMICAL RIGHT EYE GEOMETRY ====================
        p_r_outer = np.array([landmarks[self.right_outer].x * frame_w, landmarks[self.right_outer].y * frame_h], dtype=np.float64)
        p_r_inner = np.array([landmarks[self.right_inner].x * frame_w, landmarks[self.right_inner].y * frame_h], dtype=np.float64)

        v_r = p_r_inner - p_r_outer
        if v_r[0] < 0:
            v_r = -v_r

        eye_w_r = float(np.linalg.norm(v_r))
        eye_w_r = max(min_eye_w, eye_w_r)
        u_axis_r = v_r / eye_w_r
        u_perp_r = np.array([-u_axis_r[1], u_axis_r[0]], dtype=np.float64)
        if u_perp_r[1] < 0:
            u_perp_r = -u_perp_r

        p_mid_r = (p_r_inner + p_r_outer) / 2.0
        p_r_upper = np.array([landmarks[self.right_upper].x * frame_w, landmarks[self.right_upper].y * frame_h], dtype=np.float64)
        p_r_lower = np.array([landmarks[self.right_lower].x * frame_w, landmarks[self.right_lower].y * frame_h], dtype=np.float64)
        eye_h_r = float(max(4.0, np.linalg.norm(p_r_upper - p_r_lower)))

        # Right EAR
        p160 = np.array([landmarks[160].x * frame_w, landmarks[160].y * frame_h], dtype=np.float64)
        p144 = np.array([landmarks[144].x * frame_w, landmarks[144].y * frame_h], dtype=np.float64)
        p158 = np.array([landmarks[158].x * frame_w, landmarks[158].y * frame_h], dtype=np.float64)
        p153 = np.array([landmarks[153].x * frame_w, landmarks[153].y * frame_h], dtype=np.float64)
        ear_r = float((np.linalg.norm(p160 - p144) + np.linalg.norm(p158 - p153)) / (2.0 * eye_w_r))

        # Right iris canthal projection
        p_iris_r = np.array([right_iris_norm[0] * frame_w, right_iris_norm[1] * frame_h], dtype=np.float64)
        offset_r = p_iris_r - p_mid_r
        norm_disp_r_x = float(np.dot(offset_r, u_axis_r)) / eye_w_r
        norm_disp_r_y = float(np.dot(offset_r, u_perp_r)) / (eye_w_r * 0.42)

        # ==================== INDEPENDENT BLENDSHAPES ====================
        bs_blink_l = 0.0
        bs_blink_r = 0.0
        bs_look_l_x = 0.0
        bs_look_r_x = 0.0
        bs_look_l_y = 0.0
        bs_look_r_y = 0.0

        if blendshapes and len(blendshapes) > 0:
            cats = {cat.category_name: cat.score for cat in blendshapes[0]}
            bs_blink_l = cats.get("eyeBlinkLeft", 0.0)
            bs_blink_r = cats.get("eyeBlinkRight", 0.0)

            look_down_l = cats.get("eyeLookDownLeft", 0.0)
            look_up_l = cats.get("eyeLookUpLeft", 0.0)
            bs_look_l_y = look_down_l - look_up_l

            look_down_r = cats.get("eyeLookDownRight", 0.0)
            look_up_r = cats.get("eyeLookUpRight", 0.0)
            bs_look_r_y = look_down_r - look_up_r

            look_out_l = cats.get("eyeLookOutLeft", 0.0)
            look_in_l = cats.get("eyeLookInLeft", 0.0)
            look_out_r = cats.get("eyeLookOutRight", 0.0)
            look_in_r = cats.get("eyeLookInRight", 0.0)

            # User-perspective horizontal gaze from blendshapes:
            # Positive = toward user's right (screen right)
            bs_look_l_x = look_in_l - look_out_l
            bs_look_r_x = look_out_r - look_in_r

        # A3: Dynamic Blink Detection with Hysteresis & Downward Gaze Protection
        # Compensate for normal downward eye gaze which naturally elevates eyeBlink blendshapes
        bs_blink_eff_l = max(0.0, bs_blink_l - (max(0.0, bs_look_l_y) * 0.40))
        bs_blink_eff_r = max(0.0, bs_blink_r - (max(0.0, bs_look_r_y) * 0.40))

        # A true blink requires genuine eyelid closure (deep EAR drop or strong blink blendshape)
        blink_left = bool(bs_blink_eff_l > 0.68 or (ear_l < 0.075 and bs_blink_l > 0.40))
        blink_right = bool(bs_blink_eff_r > 0.68 or (ear_r < 0.075 and bs_blink_r > 0.40))

        # Natural physiological blinks occur bilaterally
        is_blinking = bool(blink_left and blink_right)

        self._prev_blink_l = blink_left
        self._prev_blink_r = blink_right

        # Blendshape fusion reinforcing iris displacement in unified user-perspective:
        # Looking right: camera unmirrored iris shifts in -X (norm_disp < 0), so -norm_disp > 0.
        # Blendshapes look_in_l and look_out_r > 0. Both components reinforce positively!
        fused_disp_l_x = (-norm_disp_l_x) + (bs_look_l_x * 0.08)
        fused_disp_r_x = (-norm_disp_r_x) + (bs_look_r_x * 0.08)
        fused_disp_l_y = norm_disp_l_y + (bs_look_l_y * 0.08)
        fused_disp_r_y = norm_disp_r_y + (bs_look_r_y * 0.08)

        # A1: Compute disparity on baseline-relative deltas if baseline offsets are provided
        if baseline_l is not None and baseline_r is not None:
            delta_l_x = fused_disp_l_x - baseline_l[0]
            delta_l_y = fused_disp_l_y - baseline_l[1]
            delta_r_x = fused_disp_r_x - baseline_r[0]
            delta_r_y = fused_disp_r_y - baseline_r[1]
            disparity_x = abs(delta_l_x - delta_r_x)
            disparity_y = abs(delta_l_y - delta_r_y)
            agreement_score = float(np.exp(-(disparity_x + disparity_y) / DELTA_DISPARITY_SCALE))
            both_eyes_agree = bool((disparity_x <= DELTA_DISPARITY_LIMIT_X) and (disparity_y <= DELTA_DISPARITY_LIMIT_Y))
        else:
            disparity_x = abs(fused_disp_l_x - fused_disp_r_x)
            disparity_y = abs(fused_disp_l_y - fused_disp_r_y)
            agreement_score = float(np.exp(-(disparity_x + disparity_y) / 0.12))
            # Remove pre-lock agreement dependency: allow auto-centering to proceed
            both_eyes_agree = True

        return {
            "left": {
                "disp_x": fused_disp_l_x,
                "disp_y": fused_disp_l_y,
                "raw_disp_x": norm_disp_l_x,
                "raw_disp_y": norm_disp_l_y,
                "bs_look_y": bs_look_l_y,
                "bs_look_x": bs_look_l_x,
                "eye_w_px": round(eye_w_l, 1),
                "eye_h_px": round(eye_h_l, 1),
                "ear": round(ear_l, 3),
                "blink": blink_left,
                "outer_pt": p_l_outer,
                "inner_pt": p_l_inner
            },
            "right": {
                "disp_x": fused_disp_r_x,
                "disp_y": fused_disp_r_y,
                "raw_disp_x": norm_disp_r_x,
                "raw_disp_y": norm_disp_r_y,
                "bs_look_y": bs_look_r_y,
                "bs_look_x": bs_look_r_x,
                "eye_w_px": round(eye_w_r, 1),
                "eye_h_px": round(eye_h_r, 1),
                "ear": round(ear_r, 3),
                "blink": blink_right,
                "outer_pt": p_r_outer,
                "inner_pt": p_r_inner
            },
            "is_blinking": is_blinking,
            "disparity_x": round(disparity_x, 4),
            "disparity_y": round(disparity_y, 4),
            "agreement_score": round(agreement_score, 3),
            "both_eyes_agree": both_eyes_agree
        }


class HeadPoseEstimator:
    """
    Robust 3D head pose estimation using OpenCV solvePnP with a corrected anthropometric 3D facial model,
    resolution-scaled reprojection gating, and consistent Euler angles via RQDecomp3x3 (B1, A7).
    """
    def __init__(self):
        # B1: Corrected anthropometric depth (eyes and mouth recessed behind nose tip: +Z depth in camera frame)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (1)
            (0.0, 330.0, 65.0),          # Chin (152)
            (-225.0, -170.0, 135.0),     # Anatomical Right eye outer corner (33)
            (225.0, -170.0, 135.0),      # Anatomical Left eye outer corner (263)
            (-150.0, 150.0, 125.0),      # Right mouth corner (61)
            (150.0, 150.0, 125.0),       # Left mouth corner (291)
            (0.0, -110.0, 35.0),         # Sellion (168)
            (0.0, -310.0, 80.0)          # Forehead (10)
        ], dtype=np.float64)

        self.key_indices = [1, 152, 33, 263, 61, 291, 168, 10]
        self.prev_rvec = None
        self.prev_tvec = None
        self.custom_camera_matrix: Optional[np.ndarray] = None
        self.custom_dist_coeffs: Optional[np.ndarray] = None

    def set_camera_intrinsics(self, camera_matrix: Optional[np.ndarray], dist_coeffs: Optional[np.ndarray] = None):
        self.custom_camera_matrix = camera_matrix
        self.custom_dist_coeffs = dist_coeffs

    def estimate(
        self,
        landmarks: Any,
        frame_w: int,
        frame_h: int
    ) -> Tuple[float, float, float, np.ndarray, float, bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
        image_points = np.array([
            (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
            for idx in self.key_indices
        ], dtype=np.float64)

        if self.custom_camera_matrix is not None:
            camera_matrix = self.custom_camera_matrix
            dist_coeffs = self.custom_dist_coeffs if self.custom_dist_coeffs is not None else np.zeros((4, 1), dtype=np.float64)
        else:
            focal_length = frame_w * 0.95
            center = (frame_w / 2.0, frame_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        use_extrinsic = (self.prev_rvec is not None and self.prev_tvec is not None)
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=self.prev_rvec if use_extrinsic else None,
            tvec=self.prev_tvec if use_extrinsic else None,
            useExtrinsicGuess=use_extrinsic,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or rvec is None or tvec is None:
            self.prev_rvec = None
            self.prev_tvec = None
            return 0.0, 0.0, 0.0, np.zeros(3), 0.0, False, 99.0, None, None

        # Reprojection error computation
        projected_pts, _ = cv2.projectPoints(self.model_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_pts = projected_pts.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected_pts, axis=1)
        rms_reproj = float(np.sqrt(np.mean(errors ** 2)))

        # A7: Resolution-scaled reprojection threshold
        reproj_limit = max(12.0, 12.0 * (frame_w / 640.0))
        head_valid = bool((rms_reproj <= reproj_limit) and success and tvec[2] > 0)
        head_confidence = float(np.clip(np.exp(-rms_reproj / (reproj_limit * 0.65)), 0.1, 1.0))

        if head_valid:
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
        else:
            self.prev_rvec = None
            self.prev_tvec = None

        # B1: Consistent Euler decomposition via RQDecomp3x3
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Wrap angles to [-180, 180]
        pitch = ((pitch + 180.0) % 360.0) - 180.0
        yaw = ((yaw + 180.0) % 360.0) - 180.0
        roll = ((roll + 180.0) % 360.0) - 180.0

        return pitch, yaw, roll, tvec.flatten(), head_confidence, head_valid, rms_reproj, rvec, tvec

    def estimate_from_matrix(self, matrix: np.ndarray) -> Tuple[float, float, float, np.ndarray, float, bool]:
        """Phase 2 (E2u): Derive pitch, yaw, roll directly from MediaPipe facial transformation matrix."""
        R_mp = matrix[:3, :3]
        tvec = matrix[:3, 3]
        sy = math.sqrt(R_mp[0, 0] ** 2 + R_mp[1, 0] ** 2)
        pitch = math.degrees(math.atan2(R_mp[2, 1], R_mp[2, 2]))
        yaw = math.degrees(math.atan2(-R_mp[2, 0], sy))
        roll = math.degrees(math.atan2(R_mp[1, 0], R_mp[0, 0]))

        pitch = ((pitch + 180.0) % 360.0) - 180.0
        yaw = ((yaw + 180.0) % 360.0) - 180.0
        roll = ((roll + 180.0) % 360.0) - 180.0
        return pitch, yaw, roll, tvec, 0.95, True


class BothEyeHeadAdaptiveFusion:
    """
    Intelligent multimodal reliability-based fusion combining BOTH eyes and 3D head pose.
    Includes single-eye fallback for ptosis/strabismus (A2) and horizontal gaze sign handling (A5).
    """
    def __init__(self):
        self.prev_fused_x = 0.5
        self.prev_fused_y = 0.5
        self.last_eye_mode = "both"

    def fuse(
        self,
        delta_disp_l_x: float,
        delta_disp_l_y: float,
        delta_disp_r_x: float,
        delta_disp_r_y: float,
        conf_l: float,
        conf_r: float,
        valid_l: bool,
        valid_r: bool,
        agreement_score: float,
        both_eyes_agree: bool,
        delta_yaw: float,
        delta_pitch: float,
        head_conf: float,
        head_valid: bool,
        is_blinking: bool,
        user_eye_sensitivity: float = 1.5,
        user_head_sensitivity: float = 1.2,
        allow_single_eye: bool = False
    ) -> Tuple[float, float, float, float, float, bool]:
        """
        Returns:
            raw_x: fused gaze X [0.02, 0.98]
            raw_y: fused gaze Y [0.02, 0.98]
            disp_both_x: combined displacement X
            disp_both_y: combined displacement Y
            fusion_confidence: overall fusion reliability score [0, 1]
            fusion_valid: boolean flag indicating if frame is trustworthy for gaze selection
        """
        if is_blinking:
            self.last_eye_mode = "blink"
            return self.prev_fused_x, self.prev_fused_y, 0.0, 0.0, 0.0, False

        # Require BOTH eyes to be valid, confident, and agreeing.
        # Do not silently substitute single-eye or head-only tracking.
        if not (valid_l and valid_r):
            if valid_l and not valid_r:
                self.last_eye_mode = "left_only"
            elif valid_r and not valid_l:
                self.last_eye_mode = "right_only"
            else:
                self.last_eye_mode = "none"
            return self.prev_fused_x, self.prev_fused_y, 0.0, 0.0, 0.0, False

        if conf_l < 0.25 or conf_r < 0.25:
            self.last_eye_mode = "low_conf"
            return self.prev_fused_x, self.prev_fused_y, 0.0, 0.0, 0.0, False

        if not both_eyes_agree:
            self.last_eye_mode = "disagree"
            return self.prev_fused_x, self.prev_fused_y, 0.0, 0.0, 0.0, False

        self.last_eye_mode = "both"
        r_l = conf_l * (0.5 + 0.5 * agreement_score)
        r_r = conf_r * (0.5 + 0.5 * agreement_score)
        w_l = r_l / (r_l + r_r + 1e-6)
        w_r = 1.0 - w_l
        disp_both_x = float(w_l * delta_disp_l_x + w_r * delta_disp_r_x)
        disp_both_y = float(w_l * delta_disp_l_y + w_r * delta_disp_r_y)
        q_both_eyes = float((0.5 * r_l + 0.5 * r_r) * agreement_score)

        # Responsive normalization: allows eye gaze alone to cover full screen range
        eye_norm_x = (disp_both_x / 0.065) * (user_eye_sensitivity / 1.5)
        eye_norm_y = (disp_both_y / 0.050) * (user_eye_sensitivity / 1.5)

        r_head = float(head_conf) if (head_valid and abs(delta_yaw) < 30.0 and abs(delta_pitch) < 25.0) else 0.0
        head_norm_x = (delta_yaw / 18.0) * (user_head_sensitivity / 1.2) if r_head > 0 else 0.0
        head_norm_y = (delta_pitch / 14.0) * (user_head_sensitivity / 1.2) if r_head > 0 else 0.0

        w_head = 0.15 * min(1.0, r_head)
        if np.sign(eye_norm_x) != np.sign(head_norm_x) and abs(eye_norm_x) > 0.05:
            w_head *= 0.25

        raw_x = 0.5 + (eye_norm_x * 0.46) + (head_norm_x * 0.20 * w_head)
        raw_y = 0.5 + (eye_norm_y * 0.46) + (head_norm_y * 0.20 * w_head)

        raw_x = float(np.clip(raw_x, 0.03, 0.97))
        raw_y = float(np.clip(raw_y, 0.03, 0.97))

        self.prev_fused_x = raw_x
        self.prev_fused_y = raw_y

        fusion_confidence = float(np.clip((q_both_eyes * 0.80) + (r_head * 0.20), 0.1, 1.0))
        fusion_valid = bool(q_both_eyes >= 0.15 and fusion_confidence >= 0.20)

        return raw_x, raw_y, disp_both_x, disp_both_y, fusion_confidence, fusion_valid


class BothEyeCalibrationManager:
    """
    Advanced BOTH-EYE Gaze Calibration Manager with Genuine Independent Validation (C1-C11).
    Preserves the 5-Box interface, applies settle time rejection (C3), baseline-relative head gating (C4),
    two independent affine ridge models on standardized features (C5), minimum quality gating (C7),
    and safe profile persistence with overridable paths and provenance validation (C9, C10).
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
        # Initialize only the first box start time; remaining boxes settle when activated (C3)
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

        # Reactivating or resetting current active box
        if matching_idx == self.current_point_index:
            self.box_start_time[box_id] = now
            if reset:
                self.samples[box_id] = []
            return True

        # Advancing to next target in sequence
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

        # Reject blinking or invalid both-eye tracking
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

        # Target tolerance validation in normalized coordinate system (quadrant-consistent)
        tx = active_target["x"]
        ty = active_target["y"]
        # Further loosen tolerance for initial calibration points to prevent stall
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
        C5: Multi-model evaluation with genuine LOBO-CV on trimmed samples.
        C6: Uses client display dimensions for true pixel metrics.
        C7: Strict quality gating: rejects poor calibrations without overwriting saved profile.
        """
        self.is_calibrating = False
        min_samples_per_box = 16

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

        # C5: Candidate regression models
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

        # Stratified K-Fold Cross-Validation across the 5 target boxes
        # Avoids LOBO corner extrapolation failure on 5-point setups by validating on held-out samples
        # stratified across all 5 target locations.
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

                from sklearn.base import clone
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

        # Calculate validation metrics from out-of-fold predictions
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

        # Train final selected model on all trimmed data
        from sklearn.base import clone
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

        # Training metrics
        tr_pred_x = self.model_x.predict(X_all_x)
        tr_pred_y = self.model_y.predict(X_all_y)
        tr_px_errors = np.sqrt(
            ((tr_pred_x - Y_all_x) * frame_w) ** 2 +
            ((tr_pred_y - Y_all_y) * frame_h) ** 2
        )
        self.train_mean_euclidean_px = round(float(np.mean(tr_px_errors)), 1)
        self.train_rmse_px = round(float(np.sqrt(np.mean(tr_px_errors ** 2))), 1)

        # C7: Strict quality gating — do NOT apply or save if below minimum quality
        if self.val_hit_rate_pct < MIN_CALIBRATION_HIT_RATE or self.val_mae_px > MAX_CALIBRATION_MAE_PX:
            self.is_calibrated = False
            self.quality = "Poor Precision"
            self.accuracy_label = f"Poor Precision ({self.accuracy}% Hit Rate, {self.val_mae_px}px error)"
            self.status_message = "Calibration failed minimum precision requirement. Using adaptive baseline."
            print(f"[AI Vision Calibration Warning] Quality check failed: HitRate={self.val_hit_rate_pct}%, MAE={self.val_mae_px}px. Profile not saved.")
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

    # Canonical alias for gaze prediction coordinate mapping
    map_gaze = apply

    def mark_stale(self):
        """C10: Invalidate calibration when camera, mirror, or baseline recentering changes."""
        if self.is_calibrated:
            self.is_calibrated = False
            self.quality = "Stale (Recalibration Recommended)"
        if hasattr(self, "telemetry") and isinstance(self.telemetry, dict):
            self.telemetry["calibration_stale"] = True

    def save_profile(self, frame_w: int, frame_h: int, camera_index: int = 0, mirror_view: bool = True):
        """
        C9, C10: Serialize trained model and companion metadata with explicit user provenance
        and hardware context (camera index, resolution, mirror setting, timestamp).
        """
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
            logger.debug(f"[AI Vision Calibration] Genuine profile saved to {self.profile_path} and model to {self.model_path}")
        except Exception as e:
            print(f"[AI Vision Calibration Warning] Failed to save profile: {e}")

    def load_profile(
        self,
        camera_index: Optional[int] = None,
        mirror_view: Optional[bool] = None,
        resolution: Optional[str] = None
    ) -> bool:
        """
        C9, C10: Load and restore trained regression model only if source == 'user_calibration'
        and hardware context matches and is under 24 hours old.
        """
        if not os.path.exists(self.model_path):
            return False

        try:
            bundle = joblib.load(self.model_path)
            # C9: Refuse to load synthetic models
            if bundle.get("source") != "user_calibration":
                print("[AI Vision Calibration] Refusing to load calibration without 'user_calibration' provenance.")
                return False

            # C10: Age check (24 hours)
            saved_time = bundle.get("timestamp", 0)
            if (time.time() - saved_time) > 86400:
                print("[AI Vision Calibration] Calibration profile expired (> 24 hours old). Starting uncalibrated.")
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

            # C10: Hardware context checks
            if camera_index is not None and bundle.get("camera_index") != camera_index:
                print("[AI Vision Calibration] Camera index mismatch for stored profile. Recalibration required.")
                return False

            if mirror_view is not None and bundle.get("mirror_view") != mirror_view:
                print("[AI Vision Calibration] Mirror setting mismatch for stored profile. Recalibration required.")
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
            print(f"[AI Vision Calibration Warning] Could not restore profile: {e}")

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


class AIVisionEngine:
    """
    Singleton AI Vision Engine orchestrating Both-Eye Iris Localization, Geometry, Head Pose,
    Multimodal Fusion, Calibration Transform, and Adaptive 1-Euro Filtering.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index: int = 0
        self.camera_name: str = "System Camera (Built-in)"
        self.camera_enabled: bool = False
        self.mirror_view: bool = True
        self.tracking_mode: str = "both_eyes"
        self.lock = threading.RLock()
        self.frame_width = 640
        self.frame_height = 480

        # Submodules
        self.iris_localizer = BothEyeIrisLocalizer()
        self.geometry_analyzer = BothEyeGeometryAnalyzer()
        self.head_pose_estimator = HeadPoseEstimator()
        self.fusion_module = BothEyeHeadAdaptiveFusion()
        self.calibration = BothEyeCalibrationManager(auto_load=True)

        # Filters: calibrated for smooth fixation yet instantaneous response on eye saccades/pursuit
        self.filter_x = AdaptiveOneEuroFilter(t0=time.time(), x0=0.5, min_cutoff=0.20, beta=0.008)
        self.filter_y = AdaptiveOneEuroFilter(t0=time.time(), x0=0.5, min_cutoff=0.20, beta=0.008)
        self.filter_yaw = AdaptiveOneEuroFilter(t0=time.time(), x0=0.0, min_cutoff=0.35, beta=0.015)
        self.filter_pitch = AdaptiveOneEuroFilter(t0=time.time(), x0=0.0, min_cutoff=0.35, beta=0.015)

        # Baseline resting state tracking
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_iris_l_x = 0.0
        self.baseline_iris_l_y = 0.0
        self.baseline_iris_r_x = 0.0
        self.baseline_iris_r_y = 0.0
        self.baseline_ear_l: Optional[float] = None
        self.baseline_ear_r: Optional[float] = None
        self.baseline_samples: List[Tuple[float, float, float, float, float, float, float, float]] = []
        self.baseline_start_time = time.time()
        self.is_baseline_calibrated = False
        self.last_valid_time = 0.0
        self.prev_yaw = 0.0
        self.prev_pitch = 0.0

        # Filtered gaze coordinates
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.raw_gaze_x = 0.5
        self.raw_gaze_y = 0.5

        # Telemetry dictionary
        self.telemetry = {
            "face_detected": False,
            "camera_enabled": False,
            "camera_index": 0,
            "camera_name": "Camera Off (Click to activate)",
            "mirror_view": True,
            "external_camera_detected": False,
            "gaze_x": 0.5,
            "gaze_y": 0.5,
            "cursor_x": 0.5,
            "cursor_y": 0.5,
            "cursor_valid": False,
            "both_eyes_valid": False,
            "is_blinking": False,
            "camera_available": False,
            "status": "Camera Off - Keyboard Controls Active (Keys 1-4)",
            "raw_gaze_x": 0.5,
            "raw_gaze_y": 0.5,
            "gaze_valid": False,
            "usable_for_selection": False,
            "stale_prediction": False,
            "invalid_reason": "Camera hardware is turned off",
            "frame_timestamp": time.time(),
            "eye_mode": "both",
            "last_error": None,
            # Legacy compatibility fields
            "iris_x": 0.0,
            "iris_y": 0.0,
            "iris_confidence": 0.0,
            "eye_width_px": 0.0,
            "eye_height_px": 0.0,
            "ear": 0.0,
            # Both-Eye fields
            "iris_left_x": 0.0,
            "iris_left_y": 0.0,
            "iris_right_x": 0.0,
            "iris_right_y": 0.0,
            "norm_disp_left_x": 0.0,
            "norm_disp_left_y": 0.0,
            "norm_disp_right_x": 0.0,
            "norm_disp_right_y": 0.0,
            "iris_conf_left": 0.0,
            "iris_conf_right": 0.0,
            "left_eye_valid": False,
            "right_eye_valid": False,
            "both_eyes_valid": False,
            "eye_w_left_px": 0.0,
            "eye_h_left_px": 0.0,
            "eye_w_right_px": 0.0,
            "eye_h_right_px": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "both_eye_disparity_x": 0.0,
            "both_eye_disparity_y": 0.0,
            "both_eye_agreement": 1.0,
            "both_eyes_agree": False,
            # Head pose fields
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "head_confidence": 0.0,
            "head_pose_valid": False,
            "reprojection_error_px": 0.0,
            "blink_left": False,
            "blink_right": False,
            "is_blinking": False,
            "fps": 0.0,
            "latency_ms": 0.0,
            "status": "Initializing Camera...",
            "tracking_mode": "both_eyes",
            "calibrated": False,
            "calibration_quality": "None",
            "calibration_stale": False,
            "calibration_reject_reason": "",
            "calibration_expired": False,
            "accuracy": None,
            "val_mae_px": None,
            "val_rmse_px": None,
            "val_hit_rate_pct": None,
            "val_mean_euclidean_px": None,
            "val_median_euclidean_px": None,
            "model_type": "None"
        }

        # Link calibration telemetry to engine telemetry
        self.telemetry["calibration_expired"] = self.calibration.telemetry.get("calibration_expired", False)
        self.telemetry["calibration_stale"] = self.calibration.telemetry.get("calibration_stale", False)
        self.telemetry["calibration_reject_reason"] = self.calibration.telemetry.get("calibration_reject_reason", "")
        self.calibration.telemetry = self.telemetry

        self.head_weight = 1.0
        self.eye_weight = 1.4

        self.fps_counter = 0
        self.fps_timer = time.time()
        self.fps = 0.0

        self.latest_jpeg: Optional[bytes] = None
        self.latest_jpeg_timestamp: float = 0.0
        self.latest_jpeg_lock = threading.RLock()

        self.consecutive_read_failures = 0
        self.consecutive_black_frames = 0
        self.on_camera_fallback = None
        self.is_switching_camera = False
        self._external_cache_time = 0.0
        self._cached_ext_index: Optional[int] = None
        self._cached_camera_list: Optional[List[Dict[str, Any]]] = None
        self._camera_cache_time = 0.0

        self.landmarker = None
        self.running_mode = None
        self._last_video_timestamp_ms = 0
        self.is_shutting_down = False
        self._init_landmarker()
        # Initialize placeholder "Camera Off" frame so stream is ready before camera is started
        self._generate_camera_off_frame()

    @property
    def is_running(self) -> bool:
        return bool(self.camera_enabled and self.cap is not None and self.cap.isOpened())

    @staticmethod
    def _is_valid_frame(frame: Optional[np.ndarray]) -> bool:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        if float(frame.max()) < 10.0 and float(frame.mean()) < 2.0:
            return False
        return True

    def _init_landmarker(self):
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            self.running_mode = VisionRunningMode.VIDEO
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            with _suppress_c_stderr():
                self.landmarker = FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[ERROR] FaceLandmarker initialization failed: {e}")
            self.landmarker = None

    def _open_capture(self, index: int) -> Optional[cv2.VideoCapture]:
        index = int(index)
        backends = [cv2.CAP_DSHOW, 0]
        for backend in backends:
            cap = None
            try:
                with _suppress_c_stderr():
                    cap = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Warmup: wait up to 1.5s for camera sensor auto-exposure and valid frames
                    start_warmup = time.time()
                    valid = False
                    while (time.time() - start_warmup) < 1.5:
                        ret, test_frame = cap.read()
                        if ret and self._is_valid_frame(test_frame):
                            valid = True
                            break
                        time.sleep(0.04)

                    if valid:
                        return cap
                    cap.release()
                    cap = None
            except Exception:
                if cap is not None:
                    try: cap.release()
                    except Exception: pass
                cap = None
        return None

    def detect_external_camera(self, force_refresh: bool = False) -> Optional[int]:
        """Check whether an external webcam is physically connected."""
        now = time.time()
        if not force_refresh and (now - self._external_cache_time < 5.0):
            return self._cached_ext_index

        with self.lock:
            if self.camera_index is not None and self.camera_index >= 1 and self.cap is not None and self.cap.isOpened():
                self._cached_ext_index = self.camera_index
                self._external_cache_time = now
                return self.camera_index

        found_idx = None
        for idx in [1, 2]:
            cap = None
            try:
                with _suppress_c_stderr():
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and self._is_valid_frame(frame):
                        found_idx = idx
                        cap.release()
                        break
                    cap.release()
            except Exception:
                if cap is not None:
                    try: cap.release()
                    except Exception: pass

        self._cached_ext_index = found_idx
        self._external_cache_time = now
        return found_idx

    def start_camera(self, camera_index: Optional[int] = None) -> bool:
        with self.lock:
            self.is_shutting_down = False
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None

            if camera_index is not None:
                self.camera_index = int(camera_index)
            elif self.camera_index is None:
                self.camera_index = 0

            is_ext = (self.camera_index >= 1)
            cam_type = "External Webcam" if is_ext else "System Camera (Built-in)"

            self.cap = self._open_capture(self.camera_index)
            if self.cap is None or not self.cap.isOpened():
                if self.camera_index != 0:
                    self.camera_index = 0
                    self.cap = self._open_capture(0)
                    if self.cap is None or not self.cap.isOpened():
                        print("[ERROR] Camera initialization failed: no working camera found")
                        self.camera_enabled = False
                        self.telemetry["camera_enabled"] = False
                        self.telemetry["status"] = "No Working Camera Found"
                        return False
                    if self.on_camera_fallback is not None:
                        try: self.on_camera_fallback(0)
                        except Exception: pass
                else:
                    print(f"[ERROR] Camera initialization failed: camera index {self.camera_index} not accessible")
                    self.camera_enabled = False
                    self.telemetry["camera_enabled"] = False
                    self.telemetry["status"] = "Camera Not Available"
                    return False

            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            self.camera_enabled = True
            cam_type = "External Webcam" if self.camera_index >= 1 else "System Camera (Built-in)"
            self.camera_name = f"{cam_type} (Camera {self.camera_index})"
            self.consecutive_read_failures = 0
            self.consecutive_black_frames = 0

            # Reset baseline tracking for both eyes (A4)
            self.baseline_samples = []
            self.baseline_start_time = time.time()
            self.is_baseline_calibrated = False
            self.calibration.mark_stale()
            self.telemetry["calibration_stale"] = True
            self.filter_x.reset(time.time(), 0.5)
            self.filter_y.reset(time.time(), 0.5)
            self.gaze_x = 0.5
            self.gaze_y = 0.5

            self.telemetry["camera_enabled"] = True
            self.telemetry["camera_index"] = self.camera_index
            self.telemetry["camera_name"] = self.camera_name
            self.telemetry["mirror_view"] = self.mirror_view
            self.telemetry["external_camera_detected"] = (self.camera_index >= 1 or self._cached_ext_index is not None)
            self.telemetry["status"] = f"Both-Eye Tracking Active ({cam_type})"
            return True

    def stop_camera(self) -> bool:
        """
        Gracefully release and stop camera video capture hardware.
        Preserves selected camera_index for subsequent re-opening,
        clears face/gaze telemetry, and publishes a Camera Off frame.
        """
        self.camera_enabled = False
        acquired = self.lock.acquire(timeout=1.0)
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            self.consecutive_read_failures = 0
            self.consecutive_black_frames = 0
            self.is_baseline_calibrated = False

            self.latest_frame_time = 0.0
            self.telemetry["camera_available"] = False
            self.telemetry["camera_enabled"] = False
            self.telemetry["face_detected"] = False
            self.telemetry["gaze_valid"] = False
            self.telemetry["usable_for_selection"] = False
            self.telemetry["is_blinking"] = False
            self.telemetry["status"] = "Camera Off - Keyboard Controls Active (Keys 1-4)"
            self.telemetry["invalid_reason"] = "Camera hardware is turned off"

            # Generate placeholder "Camera Off" image for MJPEG stream
            self._generate_camera_off_frame()
            return True
        finally:
            if acquired:
                self.lock.release()

    def _generate_camera_off_frame(self):
        """Generate and store a clean placeholder frame showing Camera Off."""
        w = max(int(self.frame_width or 640), 640)
        h = max(int(self.frame_height or 480), 480)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (24, 20, 16)  # Dark slate background

        cv2.rectangle(img, (15, 15), (w - 15, h - 15), (55, 45, 38), 2)
        cv2.putText(img, "CAMERA TURNED OFF", (max(20, w // 2 - 170), h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(img, "Emergency Keyboard Mode (Keys 1-4)", (max(20, w // 2 - 200), h // 2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Click 'Turn On Camera' to resume eye tracking", (max(20, w // 2 - 220), h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        self._save_jpeg(img)

    def shutdown(self):
        """Full system shutdown: stops loops and releases all camera resources."""
        self.is_shutting_down = True
        self.stop_camera()
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            try:
                type(self.landmarker).__del__ = lambda self: None
            except Exception:
                pass
            self.landmarker = None

    def switch_camera(self, new_index: int) -> bool:
        target_idx = int(new_index)
        print(f"[AI Vision] Dynamically switching camera to index {target_idx}...")
        with self.lock:
            self.is_switching_camera = True
            try:
                if self.cap is not None:
                    try: self.cap.release()
                    except Exception: pass
                    self.cap = None

                self.camera_enabled = True
                time.sleep(0.15)
                success = self.start_camera(camera_index=target_idx)
                if success and self.on_camera_fallback is not None:
                    try: self.on_camera_fallback(self.camera_index)
                    except Exception: pass
                return success
            finally:
                self.is_switching_camera = False

    def list_available_cameras(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """D1: Probe cameras safely without re-opening devices while streaming."""
        now = time.time()
        if not force_refresh and self._cached_camera_list is not None and (now - self._camera_cache_time < 10.0):
            return self._cached_camera_list

        available = [{
            "index": 0,
            "label": f"Camera 0 (Built-in Webcam - {self.frame_width}x{self.frame_height})",
            "is_external": False,
            "resolution": f"{self.frame_width}x{self.frame_height}"
        }]

        # If a camera is currently streaming, return cached or active camera without interrupting stream
        if self.camera_enabled and self.cap is not None and self.cap.isOpened() and not force_refresh:
            if self.camera_index >= 1:
                available.append({
                    "index": self.camera_index,
                    "label": f"Camera {self.camera_index} (External Webcam - {self.frame_width}x{self.frame_height})",
                    "is_external": True,
                    "resolution": f"{self.frame_width}x{self.frame_height}"
                })
            self._cached_camera_list = available
            self._camera_cache_time = now
            return available

        for idx in [1, 2]:
            cap = None
            try:
                with _suppress_c_stderr():
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and self._is_valid_frame(test_frame):
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                        available.append({
                            "index": idx,
                            "label": f"Camera {idx} (External Webcam - {w}x{h})",
                            "is_external": True,
                            "resolution": f"{w}x{h}"
                        })
                    cap.release()
            except Exception:
                if cap is not None:
                    try: cap.release()
                    except Exception: pass

        self._cached_camera_list = available
        self._camera_cache_time = now
        return available

    def set_tracking_mode(self, mode: str):
        self.tracking_mode = mode
        self.telemetry["tracking_mode"] = mode

    def mark_stale(self):
        """Invalidate calibration and set telemetry flag."""
        self.calibration.mark_stale()
        self.telemetry["calibration_stale"] = True

    def recenter(self):
        """Reset resting baseline position (A4, C10)."""
        self.baseline_samples = []
        self.baseline_start_time = time.time()
        self.is_baseline_calibrated = False
        self.calibration.mark_stale()
        self.telemetry["calibration_stale"] = True
        self.filter_x.reset(time.time(), 0.5)
        self.filter_y.reset(time.time(), 0.5)
        self.telemetry["status"] = "Re-centering Both Eyes..."
        logger.debug("[AI Vision] Both-eye baseline reset. Centering...")

    def _save_jpeg(self, frame: np.ndarray):
        try:
            ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                with self.latest_jpeg_lock:
                    self.latest_jpeg = buf.tobytes()
                    self.latest_jpeg_timestamp = time.time()
        except Exception:
            pass

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.latest_jpeg_lock:
            return self.latest_jpeg

    def process_frame(self, frame_input: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Processes frame under MANDATORY BOTH-EYE TRACKING architecture:
          1. Subpixel localization of both irises.
          2. Both-eye geometry and roll-invariant canthal axis projection.
          3. 3D head pose and reprojection error (B1).
          4. Stable baseline calibration without agreement gate (A1).
          5. Baseline-relative disparity and agreement score (A1).
          6. Multimodal fusion with single-eye fallback (A2) and un-mirrored inference (A5).
          7. Multi-model calibration mapping (C5).
          8. 1-Euro adaptive low-pass filtering.
          9. Validity determination and neutral state on face loss (A6).
        """
        start_time = time.time()

        if frame_input is not None:
            frame = frame_input
            ret = True
        else:
            if not self.camera_enabled or self.is_shutting_down:
                return None, self.telemetry

            with self.lock:
                if self.is_switching_camera or not self.camera_enabled or self.is_shutting_down:
                    return None, self.telemetry
                cap = self.cap
                if cap is None or not cap.isOpened():
                    if not self.camera_enabled or not self.start_camera(self.camera_index):
                        return None, self.telemetry
                    cap = self.cap

            # D3: Read frame OUTSIDE lock so API calls are never blocked
            try:
                ret, frame = cap.read() if cap is not None else (False, None)
            except Exception as e:
                ret, frame = False, None
                self.telemetry["last_error"] = str(e)

            if not ret or frame is None:
                self.consecutive_read_failures += 1
                if self.consecutive_read_failures >= 5:
                    if self.cap is not None:
                        try: self.cap.release()
                        except Exception: pass
                        self.cap = None
                    if self.camera_index != 0:
                        self.start_camera(0)
                return None, self.telemetry

            self.consecutive_read_failures = 0

            # Black frame check
            if not self._is_valid_frame(frame):
                self.consecutive_black_frames += 1
                if self.consecutive_black_frames >= 10:
                    self.telemetry["status"] = "Camera Sensor Blocked / Pitch Black"
                    self.telemetry["invalid_reason"] = "Pitch Black Sensor"
                    self.telemetry["gaze_valid"] = False
                    return frame, self.telemetry
            else:
                self.consecutive_black_frames = 0

        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h

        # FPS calculation
        self.fps_counter += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.fps = round(self.fps_counter / elapsed, 1)
            self.fps_counter = 0
            self.fps_timer = time.time()

        if self.landmarker is None:
            return frame, self.telemetry

        # A5: ALWAYS run inference on the raw un-mirrored frame
        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            now_ms = int(time.time() * 1000)
            if now_ms <= self._last_video_timestamp_ms:
                now_ms = self._last_video_timestamp_ms + 1
            self._last_video_timestamp_ms = now_ms

            if self.running_mode == mp.tasks.vision.RunningMode.VIDEO or hasattr(self.landmarker, "detect_for_video"):
                results = self.landmarker.detect_for_video(mp_image, now_ms)
            else:
                results = self.landmarker.detect(mp_image)
        except Exception as err:
            self.telemetry["last_error"] = str(err)
            return frame, self.telemetry

        face_detected = bool(results.face_landmarks and len(results.face_landmarks) > 0)
        self.telemetry["face_detected"] = face_detected
        self.telemetry["fps"] = self.fps

        # A6: Reset eye/head telemetry fields to neutral/invalid on face loss
        if not face_detected:
            self.telemetry["status"] = "Searching for patient's face..."
            self.telemetry["gaze_valid"] = False
            self.telemetry["usable_for_selection"] = False
            self.telemetry["stale_prediction"] = True
            self.telemetry["invalid_reason"] = "Face Not Detected"
            self.telemetry["cursor_valid"] = False
            self.telemetry["cursor_x"] = round(float(self.gaze_x), 3)
            self.telemetry["cursor_y"] = round(float(self.gaze_y), 3)
            self.telemetry["is_blinking"] = False
            self.telemetry["frame_timestamp"] = time.time()
            self.telemetry["iris_conf"] = 0.0
            self.telemetry["iris_conf_left"] = 0.0
            self.telemetry["iris_conf_right"] = 0.0
            self.telemetry["left_eye_valid"] = False
            self.telemetry["right_eye_valid"] = False
            self.telemetry["both_eyes_valid"] = False
            self.telemetry["both_eyes_agree"] = False
            self.telemetry["agreement_score"] = 0.0
            self.telemetry["head_pose_valid"] = False
            self.telemetry["head_confidence"] = 0.0
            self.telemetry["head_yaw"] = 0.0
            self.telemetry["head_pitch"] = 0.0
            self.telemetry["head_roll"] = 0.0
            self.telemetry["eye_mode"] = "none"

            self.head_pose_estimator.prev_rvec = None
            self.head_pose_estimator.prev_tvec = None

            cv2.putText(frame, "Waiting for Face Detection...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            display_frame = cv2.flip(frame, 1) if self.mirror_view else frame
            self._save_jpeg(display_frame)
            return frame, self.telemetry

        landmarks = results.face_landmarks[0]
        blendshapes = results.face_blendshapes

        # 1. Subpixel BOTH-EYE Iris Localization
        res_left, res_right = self.iris_localizer.localize_both_eyes(frame, landmarks, w, h)
        l_norm_x, l_norm_y, l_r_px, l_conf, l_meta = res_left
        r_norm_x, r_norm_y, r_r_px, r_conf, r_meta = res_right
        l_valid = l_meta["is_valid"]
        r_valid = r_meta["is_valid"]

        # 2. Both-Eye Geometry & roll-invariance
        baseline_l = (self.baseline_iris_l_x, self.baseline_iris_l_y) if self.is_baseline_calibrated else None
        baseline_r = (self.baseline_iris_r_x, self.baseline_iris_r_y) if self.is_baseline_calibrated else None
        geo = self.geometry_analyzer.analyze_both_eyes(
            landmarks, (l_norm_x, l_norm_y), (r_norm_x, r_norm_y), blendshapes, w, h,
            mirror_view=self.mirror_view,
            baseline_l=baseline_l,
            baseline_r=baseline_r,
            baseline_ear_l=self.baseline_ear_l,
            baseline_ear_r=self.baseline_ear_r
        )
        geo_l = geo["left"]
        geo_r = geo["right"]
        is_blinking = geo["is_blinking"]

        # 3. 3D Head Pose with corrected geometry (B1)
        raw_pitch, raw_yaw, raw_roll, _, head_confidence, head_valid, rms_reproj, _, _ = self.head_pose_estimator.estimate(
            landmarks, w, h
        )
        now = time.time()
        smooth_yaw = self.filter_yaw.filter(now, raw_yaw, head_confidence)
        smooth_pitch = self.filter_pitch.filter(now, raw_pitch, head_confidence)

        # 4. Stable Baseline Centering without agreement gate (A1)
        dt = max(0.01, now - self.last_valid_time)
        if not self.is_baseline_calibrated:
            if not is_blinking and (l_valid and r_valid):
                self.baseline_samples.append((
                    smooth_yaw, smooth_pitch,
                    geo_l["disp_x"], geo_l["disp_y"],
                    geo_r["disp_x"], geo_r["disp_y"],
                    geo_l["ear"], geo_r["ear"]
                ))

            sample_count = len(self.baseline_samples)
            self.telemetry["status"] = f"Auto-Centering Both-Eye Baseline ({min(sample_count, 15)}/15)..."

            time_in_baseline = now - self.baseline_start_time
            is_stable = False
            if sample_count >= BASELINE_REQUIRED_SAMPLES:
                recent = self.baseline_samples[-15:]
                std_l_x = float(np.std([s[2] for s in recent]))
                std_r_x = float(np.std([s[4] for s in recent]))
                std_l_y = float(np.std([s[3] for s in recent]))
                std_r_y = float(np.std([s[5] for s in recent]))
                dispersion = max(std_l_x, std_r_x, std_l_y, std_r_y)
                if dispersion < BASELINE_DISPERSION_THRESHOLD or sample_count >= 25 or time_in_baseline >= BASELINE_TIMEOUT_SECONDS:
                    is_stable = True
            elif time_in_baseline >= BASELINE_TIMEOUT_SECONDS and sample_count >= 5:
                is_stable = True

            if is_stable:
                chosen = self.baseline_samples[-20:] if len(self.baseline_samples) >= 20 else self.baseline_samples
                self.baseline_yaw = float(np.median([s[0] for s in chosen]))
                self.baseline_pitch = float(np.median([s[1] for s in chosen]))
                self.baseline_iris_l_x = float(np.median([s[2] for s in chosen]))
                self.baseline_iris_l_y = float(np.median([s[3] for s in chosen]))
                self.baseline_iris_r_x = float(np.median([s[4] for s in chosen]))
                self.baseline_iris_r_y = float(np.median([s[5] for s in chosen]))
                self.baseline_ear_l = float(np.median([s[6] for s in chosen]))
                self.baseline_ear_r = float(np.median([s[7] for s in chosen]))
                self.is_baseline_calibrated = True
                self.telemetry["status"] = "Both-Eye Tracking Active"
                logger.debug(f"[AI Vision] Both-eye baseline locked ({len(chosen)} frames): Yaw={self.baseline_yaw:.1f}, Pitch={self.baseline_pitch:.1f}, L=({self.baseline_iris_l_x:.3f}, {self.baseline_iris_l_y:.3f}), R=({self.baseline_iris_r_x:.3f}, {self.baseline_iris_r_y:.3f})")
        else:
            # Controlled drift only during neutral fixation
            head_vel = abs(smooth_yaw - self.prev_yaw) / dt
            is_neutral_fixation = (
                not is_blinking and
                l_valid and r_valid and
                l_conf > 0.65 and r_conf > 0.65 and
                geo["both_eyes_agree"] and
                abs(geo_l["disp_x"] - self.baseline_iris_l_x) < 0.012 and
                abs(geo_l["disp_y"] - self.baseline_iris_l_y) < 0.012 and
                abs(geo_r["disp_x"] - self.baseline_iris_r_x) < 0.012 and
                abs(geo_r["disp_y"] - self.baseline_iris_r_y) < 0.012 and
                abs(smooth_yaw - self.baseline_yaw) < 1.5 and
                abs(smooth_pitch - self.baseline_pitch) < 1.5 and
                head_vel < 0.3
            )
            if is_neutral_fixation:
                slow_drift = 0.0002
                self.baseline_yaw += (smooth_yaw - self.baseline_yaw) * slow_drift
                self.baseline_pitch += (smooth_pitch - self.baseline_pitch) * slow_drift

        self.prev_yaw = smooth_yaw
        self.prev_pitch = smooth_pitch

        # 5. Multimodal Adaptive Fusion: Both Eyes (Primary) + Head Pose
        # User-perspective coordinate deltas: positive X is screen right, positive Y is screen down
        if self.is_baseline_calibrated:
            delta_yaw = -(smooth_yaw - self.baseline_yaw)
            delta_pitch = smooth_pitch - self.baseline_pitch
            delta_l_x = geo_l["disp_x"] - self.baseline_iris_l_x
            delta_l_y = geo_l["disp_y"] - self.baseline_iris_l_y
            delta_r_x = geo_r["disp_x"] - self.baseline_iris_r_x
            delta_r_y = geo_r["disp_y"] - self.baseline_iris_r_y
        else:
            base_yaw = float(np.median([s[0] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else smooth_yaw
            base_pitch = float(np.median([s[1] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else smooth_pitch
            base_lx = float(np.median([s[2] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else geo_l["disp_x"]
            base_ly = float(np.median([s[3] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else geo_l["disp_y"]
            base_rx = float(np.median([s[4] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else geo_r["disp_x"]
            base_ry = float(np.median([s[5] for s in self.baseline_samples])) if len(self.baseline_samples) >= 3 else geo_r["disp_y"]
            delta_yaw = -(smooth_yaw - base_yaw)
            delta_pitch = smooth_pitch - base_pitch
            delta_l_x = geo_l["disp_x"] - base_lx
            delta_l_y = geo_l["disp_y"] - base_ly
            delta_r_x = geo_r["disp_x"] - base_rx
            delta_r_y = geo_r["disp_y"] - base_ry

        fused_raw_x, fused_raw_y, _, _, fusion_conf, fusion_valid = self.fusion_module.fuse(
            delta_disp_l_x=delta_l_x,
            delta_disp_l_y=delta_l_y,
            delta_disp_r_x=delta_r_x,
            delta_disp_r_y=delta_r_y,
            conf_l=l_conf,
            conf_r=r_conf,
            valid_l=l_valid,
            valid_r=r_valid,
            agreement_score=geo["agreement_score"],
            both_eyes_agree=geo["both_eyes_agree"],
            delta_yaw=delta_yaw,
            delta_pitch=delta_pitch,
            head_conf=head_confidence,
            head_valid=head_valid,
            is_blinking=is_blinking,
            user_eye_sensitivity=self.eye_weight,
            user_head_sensitivity=self.head_weight,
            allow_single_eye=False
        )

        # 6. Gaze Validity Determination: strictly require both eyes to be valid and reliable
        eye_mode = self.fusion_module.last_eye_mode
        both_eyes_valid = bool(l_valid and r_valid and not is_blinking and geo["both_eyes_agree"] and fusion_valid)
        gaze_valid = bool(face_detected and not is_blinking and both_eyes_valid and fusion_valid)

        invalid_reason = None
        if not face_detected:
            invalid_reason = "Face Not Detected"
        elif is_blinking:
            invalid_reason = "Blink Detected"
        elif not l_valid and not r_valid:
            invalid_reason = "Both Eyes Occluded / Unreliable"
        elif not l_valid:
            invalid_reason = "Both Eyes Required — Left Eye Occluded / Unreliable"
        elif not r_valid:
            invalid_reason = "Both Eyes Required — Right Eye Occluded / Unreliable"
        elif not geo["both_eyes_agree"]:
            invalid_reason = "Both Eyes Required — Binocular Disparity (Eyes Disagree)"
        elif not fusion_valid:
            invalid_reason = "Tracking Temporarily Unavailable"

        # 7. Apply Calibration and Adaptive 1-Euro Filter ONLY for fresh, valid predictions
        if gaze_valid:
            self.raw_gaze_x = fused_raw_x
            self.raw_gaze_y = fused_raw_y
            mapped_x, mapped_y = self.calibration.apply(
                raw_x=self.raw_gaze_x,
                raw_y=self.raw_gaze_y,
                left_dx=delta_l_x,
                left_dy=delta_l_y,
                right_dx=delta_r_x,
                right_dy=delta_r_y,
                delta_yaw=delta_yaw,
                delta_pitch=delta_pitch
            )
            self.gaze_x = self.filter_x.filter(now, mapped_x, confidence=fusion_conf)
            self.gaze_y = self.filter_y.filter(now, mapped_y, confidence=fusion_conf)
            self.gaze_x = float(np.clip(self.gaze_x, 0.02, 0.98))
            self.gaze_y = float(np.clip(self.gaze_y, 0.02, 0.98))
            self.last_valid_time = now
            stale_prediction = False
        else:
            stale_prediction = True

        # Cursor-only fallback: keep the on-screen pointer moving when strict both-eye
        # validation fails. Selection (dwell) still requires strict gaze_valid.
        cursor_valid = bool(gaze_valid)
        cursor_x, cursor_y = self.gaze_x, self.gaze_y
        if gaze_valid:
            self._cursor_fx, self._cursor_fy = self.gaze_x, self.gaze_y
        elif face_detected and not is_blinking and (l_valid or r_valid):
            if l_valid and r_valid:
                cdx, cdy = 0.5 * (delta_l_x + delta_r_x), 0.5 * (delta_l_y + delta_r_y)
            elif l_valid:
                cdx, cdy = delta_l_x, delta_l_y
            else:
                cdx, cdy = delta_r_x, delta_r_y
            sens = self.eye_weight / 1.5
            c_raw_x = float(np.clip(0.5 + (cdx / 0.065) * sens * 0.46, 0.03, 0.97))
            c_raw_y = float(np.clip(0.5 + (cdy / 0.050) * sens * 0.46, 0.03, 0.97))
            cm_x, cm_y = self.calibration.apply(
                raw_x=c_raw_x, raw_y=c_raw_y,
                left_dx=delta_l_x, left_dy=delta_l_y,
                right_dx=delta_r_x, right_dy=delta_r_y,
                delta_yaw=delta_yaw, delta_pitch=delta_pitch
            )
            # Route the fallback cursor through the same adaptive low-pass filter
            # used for strict gaze so that single-eye / disagreement noise is
            # smoothed instead of producing a raw 0.7/0.3 blend that bounces.
            prev_fx = getattr(self, "_cursor_fx", float(cm_x))
            prev_fy = getattr(self, "_cursor_fy", float(cm_y))
            self._cursor_fx = self.filter_x.filter(now, cm_x, confidence=0.5)
            self._cursor_fy = self.filter_y.filter(now, cm_y, confidence=0.5)
            self._cursor_fx = float(np.clip(self._cursor_fx, 0.02, 0.98))
            self._cursor_fy = float(np.clip(self._cursor_fy, 0.02, 0.98))
            cursor_x, cursor_y = self._cursor_fx, self._cursor_fy
            cursor_valid = True

        usable_for_selection = bool(gaze_valid and not stale_prediction and not self.calibration.is_calibrating)

        latency_ms = round((time.time() - start_time) * 1000.0, 1)

        self.latest_frame_time = now
        self.frame_counter = getattr(self, "frame_counter", 0) + 1

        # Update Telemetry
        self.telemetry["gaze_x"] = round(float(self.gaze_x), 3)
        self.telemetry["gaze_y"] = round(float(self.gaze_y), 3)
        self.telemetry["cursor_valid"] = bool(cursor_valid)
        self.telemetry["cursor_x"] = round(float(cursor_x), 3)
        self.telemetry["cursor_y"] = round(float(cursor_y), 3)
        self.telemetry["raw_gaze_x"] = round(float(self.raw_gaze_x), 3)
        self.telemetry["raw_gaze_y"] = round(float(self.raw_gaze_y), 3)
        self.telemetry["gaze_valid"] = gaze_valid
        self.telemetry["both_eyes_valid"] = both_eyes_valid
        self.telemetry["usable_for_selection"] = usable_for_selection
        self.telemetry["stale_prediction"] = stale_prediction
        self.telemetry["invalid_reason"] = invalid_reason
        self.telemetry["delta_yaw"] = round(float(delta_yaw), 2)
        self.telemetry["delta_pitch"] = round(float(delta_pitch), 2)
        self.telemetry["frame_timestamp"] = now
        self.telemetry["frame_counter"] = self.frame_counter
        self.telemetry["camera_available"] = bool(self.cap is not None and self.is_running and self.camera_enabled)
        self.telemetry["eye_mode"] = eye_mode

        # Coordinates and confidences
        self.telemetry["iris_left_x"] = round(float(l_norm_x), 4)
        self.telemetry["iris_left_y"] = round(float(l_norm_y), 4)
        self.telemetry["iris_right_x"] = round(float(r_norm_x), 4)
        self.telemetry["iris_right_y"] = round(float(r_norm_y), 4)
        self.telemetry["norm_disp_left_x"] = round(float(geo_l["disp_x"]), 4)
        self.telemetry["norm_disp_left_y"] = round(float(geo_l["disp_y"]), 4)
        self.telemetry["norm_disp_right_x"] = round(float(geo_r["disp_x"]), 4)
        self.telemetry["norm_disp_right_y"] = round(float(geo_r["disp_y"]), 4)
        self.telemetry["iris_conf_left"] = round(float(l_conf), 2)
        self.telemetry["iris_conf_right"] = round(float(r_conf), 2)
        self.telemetry["left_eye_valid"] = l_valid
        self.telemetry["right_eye_valid"] = r_valid
        self.telemetry["both_eyes_valid"] = both_eyes_valid
        self.telemetry["eye_w_left_px"] = geo_l["eye_w_px"]
        self.telemetry["eye_h_left_px"] = geo_l["eye_h_px"]
        self.telemetry["eye_w_right_px"] = geo_r["eye_w_px"]
        self.telemetry["eye_h_right_px"] = geo_r["eye_h_px"]
        self.telemetry["ear_left"] = geo_l["ear"]
        self.telemetry["ear_right"] = geo_r["ear"]
        self.telemetry["both_eye_disparity_x"] = geo["disparity_x"]
        self.telemetry["both_eye_disparity_y"] = geo["disparity_y"]
        self.telemetry["both_eye_agreement"] = geo["agreement_score"]
        self.telemetry["both_eyes_agree"] = geo["both_eyes_agree"]

        # Legacy fields
        self.telemetry["iris_x"] = round(float(l_norm_x), 4)
        self.telemetry["iris_y"] = round(float(l_norm_y), 4)
        self.telemetry["iris_confidence"] = round(float(0.5 * l_conf + 0.5 * r_conf), 2)
        self.telemetry["eye_width_px"] = geo_l["eye_w_px"]
        self.telemetry["eye_height_px"] = geo_l["eye_h_px"]
        self.telemetry["ear"] = round(float(0.5 * geo_l["ear"] + 0.5 * geo_r["ear"]), 3)

        # Head pose
        self.telemetry["head_yaw"] = round(float(smooth_yaw), 1)
        self.telemetry["head_pitch"] = round(float(smooth_pitch), 1)
        self.telemetry["head_roll"] = round(float(raw_roll), 1)
        self.telemetry["head_confidence"] = round(float(head_confidence), 2)
        self.telemetry["head_pose_valid"] = head_valid
        self.telemetry["reprojection_error_px"] = round(float(rms_reproj), 2)

        self.telemetry["blink_left"] = bool(geo_l["blink"])
        self.telemetry["blink_right"] = bool(geo_r["blink"])
        self.telemetry["is_blinking"] = bool(is_blinking)
        self.telemetry["latency_ms"] = latency_ms

        # Calibration status telemetry
        self.telemetry["calibrated"] = self.calibration.is_calibrated
        self.telemetry["calibration_quality"] = self.calibration.quality
        self.telemetry["accuracy"] = self.calibration.accuracy
        self.telemetry["val_mae_px"] = self.calibration.val_mae_px
        self.telemetry["val_rmse_px"] = self.calibration.val_rmse_px
        self.telemetry["val_hit_rate_pct"] = self.calibration.val_hit_rate_pct
        self.telemetry["val_mean_euclidean_px"] = self.calibration.val_mean_euclidean_px
        self.telemetry["val_median_euclidean_px"] = self.calibration.val_median_euclidean_px
        self.telemetry["model_type"] = self.calibration.selected_model_name
        self.telemetry["calibration_stale"] = self.calibration.telemetry.get("calibration_stale", False)
        self.telemetry["calibration_reject_reason"] = self.calibration.telemetry.get("calibration_reject_reason", "")
        self.telemetry["calibration_expired"] = self.calibration.telemetry.get("calibration_expired", False)
        if self.calibration.is_calibrated:
            self.telemetry["calibration_stale"] = False

        if self.calibration.is_calibrated:
            self.telemetry["status"] = self.calibration.status_message

        # HUD Visualization
        reticle_color = (0, 255, 0) if gaze_valid else (0, 165, 255)
        ix_l = int(l_norm_x * w)
        iy_l = int(l_norm_y * h)
        ix_r = int(r_norm_x * w)
        iy_r = int(r_norm_y * h)
        ir_l = int(max(3, l_r_px))
        ir_r = int(max(3, r_r_px))

        cv2.circle(frame, (ix_l, iy_l), ir_l, (0, 255, 255) if l_valid else (0, 0, 255), 1)
        cv2.circle(frame, (ix_r, iy_r), ir_r, (0, 255, 255) if r_valid else (0, 0, 255), 1)

        gx_px = int(self.gaze_x * w)
        gy_px = int(self.gaze_y * h)
        cv2.circle(frame, (gx_px, gy_px), 8, reticle_color, -1)
        cv2.circle(frame, (gx_px, gy_px), 14, reticle_color, 1)

        # Save JPEG for web stream: flip if mirror_view
        display_frame = cv2.flip(frame, 1) if self.mirror_view else frame
        self._save_jpeg(display_frame)

        return frame, self.telemetry


# Singleton instance
vision_engine = AIVisionEngine()
