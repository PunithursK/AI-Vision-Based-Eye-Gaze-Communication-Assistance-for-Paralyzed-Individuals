/**
 * AI-Vision Communication Assistance for Paralyzed Individuals
 * Client-Side Application Engine: Single-Eye Gaze Tracking, 5-Box Calibration (Target >=85%),
 * Camera On/Off Control, Smooth Slow Cursor Glide, Hotkey Numbers (1-8),
 * 3-Second Dwell Card Selection, Lockout Protocol, and Caregiver Notification Telemetry.
 */

class GazeAssistant {
    constructor() {
        // State variables
        this.isLocked = false;
        this.lockedBlockId = null;
        this.activeHoverBlock = null;
        this.dwellStartTime = null;
        this.dwellDuration = 3000; // 3.0s (3000ms) required dwell time
        this.maxDwellOffset = 264; // SVG stroke-dasharray

        // Raw & Smooth Gaze coordinates
        this.lastGazeX = 0.5;
        this.lastGazeY = 0.5;
        this.displayGazeX = 0.5; // Smooth interpolated coordinate for slow, calm pointer motion
        this.displayGazeY = 0.5;
        this.hasReceivedValidGaze = false;
        this.hasEverTrackedFace = false;

        // Camera power state
        this.cameraRunning = true;

        // Audio and Speech
        this.audioCtx = null;
        this.soundEnabled = true;
        this.voiceEnabled = false; // Voice assistance completely disabled per user request

        // 5-Point Calibration Protocol in Exact Required Order:
        // 1. TOP RIGHT — (0.90, 0.10)
        // 2. TOP LEFT — (0.10, 0.10)
        // 3. BOTTOM RIGHT — (0.90, 0.90)
        // 4. BOTTOM LEFT — (0.10, 0.90)
        // 5. CENTRE — (0.50, 0.50)
        this.isCalibrating = false;
        this.isCalibrated = false;
        this.currentCalibIndex = 0;
        this.calibBoxes = [
            { id: "top_right", name: "Top Right", x: 0.90, y: 0.10 },
            { id: "top_left", name: "Top Left", x: 0.10, y: 0.10 },
            { id: "bottom_right", name: "Bottom Right", x: 0.90, y: 0.90 },
            { id: "bottom_left", name: "Bottom Left", x: 0.10, y: 0.90 },
            { id: "center", name: "Centre", x: 0.50, y: 0.50 }
        ];

        // Selection State Machine (Part 5):
        // IDLE, TRACKING, DWELLING, CONFIRMED, COOLDOWN, WAITING_FOR_RELEASE, CAMERA_UNAVAILABLE
        this.selectionState = 'IDLE';
        this.lastConfirmedCardId = null;
        this.lastTelemetryReceiveTime = Date.now();
        this.isStalePrediction = false;
        this.calibBoxDuration = 2500;
        this.calibrationDwellDuration = 1500;
        this.calibDwellActive = false;
        this.calibDwellStartTime = null;
        this.calibBlinkGraceStartTime = null;
        this.calibInsideTime = 0;
        this.lastCalibFrameTime = Date.now();
        this.calibBoxTimeout = null;
        this.lastSampleSentTime = 0;
        this.lastRawGazeX = 0.5;
        this.lastRawGazeY = 0.5;
        this.acceptedSamplesInBox = 0;
        this.isSendingSample = false;
        this.lastDwellTickWindow = -1;
        this.lastTriggerSentTime = 0;
        this.autoCalibChecked = false;
        this.trackingStableStart = null;
        this.fixationStartTime = null;
        this.bothEyesValid = false;
        this.isBlinking = false;

        // Elements
        this.cursorEl = document.getElementById("gaze-cursor");
        if (this.cursorEl) {
            this.cursorEl.style.display = "flex";
            this.cursorEl.style.opacity = "0.5";
            this.cursorEl.style.pointerEvents = "none";
        }
        this.blocksGrid = document.getElementById("blocks-grid");
        this.cards = document.querySelectorAll(".request-card");
        this.lockoutModal = document.getElementById("lockout-modal");
        this.calibrationModal = document.getElementById("calibration-modal");
        this.calibrationBox = document.getElementById("calibration-box");
        this.calibrationTargets = document.querySelectorAll(".calibration-target");
        this.calibBoxFill = document.getElementById("calib-box-fill");
        this.calibBoxBadge = document.getElementById("calib-box-badge");
        this.calibBoxSublabel = document.getElementById("calib-box-sublabel");
        this.calibStepBadge = document.getElementById("calib-step-badge");
        this.calibPointName = document.getElementById("calib-point-name");
        this.calibProgressBar = document.getElementById("calib-progress-bar");
        this.calibCountdownText = document.getElementById("calib-countdown-text");
        this.calibInstruction = document.getElementById("calib-instruction");
        this.calibAccuracyPill = document.getElementById("calib-accuracy-pill");
        this.pipWindow = document.getElementById("cam-pip-window");
        this.logsModal = document.getElementById("logs-modal");
        this.settingsModal = document.getElementById("settings-modal");
        this.toastContainer = document.getElementById("toast-container");

        // Camera control elements & mirror support
        this.camPowerBtn = document.getElementById("btn-toggle-camera-power");
        this.camPowerIcon = document.getElementById("cam-power-icon");
        this.camPowerText = document.getElementById("cam-power-text");
        this.quickCamSelect = document.getElementById("quick-camera-select");
        this.cfgCamSelect = document.getElementById("cfg-camera-device");
        this.btnToggleMirror = document.getElementById("btn-toggle-mirror");
        this.mirrorBtnText = document.getElementById("mirror-btn-text");
        this.btnPipMirror = document.getElementById("btn-pip-mirror");
        this.pipCamLabel = document.getElementById("pip-cam-label");
        this.pipVideoStream = document.getElementById("pip-video-stream");

        this.cameraIndex = 0;
        this.mirrorView = true;
        this.availableCameras = [];

        // Telemetry display elements
        this.headPoseText = document.getElementById("head-pose-text");
        this.visionStatusText = document.getElementById("vision-status-text");
        this.statusDot = document.getElementById("status-dot");
        this.currentDwellLabel = document.getElementById("current-dwell-label");
        this.cooldownFill = document.getElementById("cooldown-fill");
        this.cooldownInterval = null;
        this.faceLostBanner = document.getElementById("face-lost-banner");

        // Face detection & Caregiver auth state
        this.faceDetected = false;
        this.patientSelectionAuthorized = true;
        this._calibStaleShown = false;
        this._calibExpiredShown = false;

        // Initialize components
        this.initAudio();
        this.bindEvents();
        this.fetchInitialTelemetry();
        // D1: Delay fetchCameras until calibration finishes/skips or after startup settle
        setTimeout(() => {
            if (!this.isCalibrating) this.fetchCameras();
        }, 2000);
        this.loadConfig();
        this.startSSE();
        this.startRenderLoop();

        // Cards are unlocked and ready for zero-calibration hands-free gaze tracking
        if (this.blocksGrid) {
            this.blocksGrid.classList.remove("cards-locked-for-calibration");
        }
    }

    /**
     * F5: CSRF Token & Header Helper
     */
    getCsrfToken() {
        const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
        return match ? match[1] : "";
    }

    getHeaders() {
        const token = this.getCsrfToken();
        const headers = { "Content-Type": "application/json" };
        if (token) {
            headers["X-CSRF-Token"] = token;
        }
        return headers;
    }

    /**
     * Fetch immediate instantaneous telemetry upon page load
     */
    async fetchInitialTelemetry() {
        try {
            const resp = await fetch("/api/telemetry");
            if (resp.ok) {
                const data = await resp.json();
                this.handleTelemetry(data);
            }
        } catch (e) {
            // Handled by SSE stream
        }
    }

    /**
     * Manage caregiver authorization state for patient card dwelling and selections
     */
    setPatientSelectionAuthorization(authorized) {
        this.patientSelectionAuthorized = true;
    }

    clearActivePatientCard() {
        if (this.cards) {
            this.cards.forEach(c => c.classList.remove("active", "hovered", "dwell-active"));
        }
    }

    cancelPendingPatientSelection() {
        this.resetAllBlockDwells();
    }

    /**
     * Web Audio API Synthesizer: Crystal clear chimes and feedback.
     */
    initAudio() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
        } catch (e) {
            console.warn("Web Audio API not supported", e);
        }
    }

    resumeAudio() {
        if (this.audioCtx && this.audioCtx.state === "suspended") {
            this.audioCtx.resume();
        }
    }

    playTone(freq, type = "sine", duration = 0.15, gainVal = 0.1) {
        if (!this.soundEnabled || !this.audioCtx) return;
        this.resumeAudio();
        try {
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
            gain.gain.setValueAtTime(gainVal, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(this.audioCtx.destination);
            osc.start();
            osc.stop(this.audioCtx.currentTime + duration);
        } catch (e) { }
    }

    playChimeLock() {
        if (!this.soundEnabled) return;
        this.playTone(523.25, "triangle", 0.4, 0.2); // C5
        setTimeout(() => this.playTone(659.25, "triangle", 0.45, 0.2), 100); // E5
        setTimeout(() => this.playTone(783.99, "sine", 0.6, 0.25), 200); // G5
    }

    playChimeDelivered() {
        if (!this.soundEnabled) return;
        this.playTone(659.25, "sine", 0.25, 0.15); // E5
        setTimeout(() => this.playTone(880.00, "sine", 0.5, 0.25), 120); // A5
    }

    playChimeStep() {
        if (!this.soundEnabled) return;
        this.playTone(740.00, "sine", 0.14, 0.12);
        setTimeout(() => this.playTone(980.00, "sine", 0.20, 0.18), 90);
    }

    /**
     * Bind UI action buttons and keyboard shortcuts (Numbers 1-4 for instant dispatch)
     */
    updateSimulationModeUI(isSim) {
        const modeBadge = document.getElementById("smtp-mode-badge");
        if (modeBadge) {
            modeBadge.textContent = isSim ? "Simulation Mode" : "Live Brevo";
            modeBadge.className = isSim ? "badge text-bg-warning ms-2" : "badge text-bg-success ms-2";
        }
        const modeText = document.getElementById("smtp-mode-text");
        if (modeText) {
            if (isSim) {
                modeText.className = "small text-warning";
                modeText.innerHTML = '<i class="bi bi-info-circle me-1"></i>Simulation mode — no real email is sent';
            } else {
                modeText.className = "small text-success";
                modeText.innerHTML = '<i class="bi bi-check-circle me-1"></i>Live Brevo mode — sends to configured caregiver alert email';
            }
        }
    }

    bindEvents() {
        // Camera on/off toggle button
        if (this.camPowerBtn) {
            this.camPowerBtn.addEventListener("click", () => this.toggleCameraPower());
        }

        // Camera device quick switch dropdown in header
        if (this.quickCamSelect) {
            this.quickCamSelect.addEventListener("change", (e) => {
                this.switchCamera(e.target.value);
            });
        }

        // Camera device selector in settings modal
        if (this.cfgCamSelect) {
            this.cfgCamSelect.addEventListener("change", (e) => {
                this.switchCamera(e.target.value);
            });
        }

        // Mirror view toggle buttons
        if (this.btnToggleMirror) {
            this.btnToggleMirror.addEventListener("click", () => this.toggleMirror());
        }
        if (this.btnPipMirror) {
            this.btnPipMirror.addEventListener("click", () => this.toggleMirror());
        }
        const cfgMirrorSelect = document.getElementById("cfg-mirror-view");
        if (cfgMirrorSelect) {
            cfgMirrorSelect.addEventListener("change", (e) => {
                this.toggleMirror(e.target.value === "1");
            });
        }

        // Calibration button in top navbar
        const calibBtn = document.getElementById("btn-open-calibration");
        if (calibBtn) {
            calibBtn.addEventListener("click", () => {
                const sysCtrlMenu = document.getElementById("system-controls-menu");
                if (sysCtrlMenu) sysCtrlMenu.style.display = "none";
                this.startCalibration();
            });
        }

        // Skip calibration buttons (header button and camera fallback banner)
        const skipBtn = document.getElementById("btn-skip-calibration");
        if (skipBtn) {
            skipBtn.addEventListener("click", () => this.skipCalibration());
        }
        const fallbackBtn = document.getElementById("btn-skip-fallback");
        if (fallbackBtn) {
            fallbackBtn.addEventListener("click", () => this.skipCalibration());
        }

        // Re-Center baseline gaze button & keyboard shortcut 'C'
        const recenterBtn = document.getElementById("btn-recenter");
        if (recenterBtn) {
            recenterBtn.addEventListener("click", () => this.triggerRecenter());
        }

        // System Controls Dropdown Toggle (Image 5 Compact Menu)
        const sysCtrlToggle = document.getElementById("btn-system-controls-toggle");
        const sysCtrlMenu = document.getElementById("system-controls-menu");
        const sysCtrlClose = document.getElementById("btn-close-system-controls");
        const sysCtrlChevron = document.getElementById("sys-ctrl-chevron");

        if (sysCtrlToggle && sysCtrlMenu) {
            const toggleSysMenu = (forceState) => {
                const currentlyOpen = sysCtrlMenu.style.display === "block";
                const targetOpen = forceState !== undefined ? forceState : !currentlyOpen;
                sysCtrlMenu.style.display = targetOpen ? "block" : "none";
                sysCtrlToggle.setAttribute("aria-expanded", String(targetOpen));
                if (sysCtrlChevron) {
                    sysCtrlChevron.className = targetOpen ? "bi bi-chevron-up ms-1" : "bi bi-chevron-down ms-1";
                }
            };

            sysCtrlToggle.addEventListener("click", (e) => {
                e.stopPropagation();
                toggleSysMenu();
            });

            if (sysCtrlClose) {
                sysCtrlClose.addEventListener("click", (e) => {
                    e.stopPropagation();
                    toggleSysMenu(false);
                });
            }

            document.addEventListener("click", (e) => {
                if (sysCtrlMenu.style.display === "block" && !sysCtrlMenu.contains(e.target) && !sysCtrlToggle.contains(e.target)) {
                    toggleSysMenu(false);
                }
            });

            sysCtrlMenu.addEventListener("click", (e) => {
                e.stopPropagation();
            });
        }

        // Keyboard hotkeys: Numbers 1 through 8 for instant block dispatch!
        document.addEventListener("keydown", (e) => {
            // Check if user is typing in an input field or a popup modal is open
            const isTyping = e.target && (
                e.target.tagName === "INPUT" ||
                e.target.tagName === "TEXTAREA" ||
                e.target.tagName === "SELECT" ||
                e.target.isContentEditable
            );
            const isModalOpen = (
                (this.settingsModal && !this.settingsModal.classList.contains("hidden")) ||
                (this.logsModal && !this.logsModal.classList.contains("hidden"))
            );

            // Escape closes open modals, system controls menu, or skips calibration if open
            if (e.key === "Escape") {
                const sysCtrlMenu = document.getElementById("system-controls-menu");
                if (sysCtrlMenu && sysCtrlMenu.style.display === "block") {
                    sysCtrlMenu.style.display = "none";
                    const sysCtrlChevron = document.getElementById("sys-ctrl-chevron");
                    if (sysCtrlChevron) sysCtrlChevron.className = "bi bi-chevron-down ms-1";
                    return;
                }
                if (this.isCalibrating) {
                    this.skipCalibration();
                    return;
                }
                this.closeAllModals();
                return;
            }

            // CRITICAL REQUIREMENT: If Caregiver Settings or Logs popup is open or user is typing,
            // completely disable block number triggers and hotkeys so numbers can be typed safely!
            const isTextEntry = e.target && (
                e.target.tagName === "TEXTAREA" ||
                e.target.isContentEditable ||
                (e.target.tagName === "INPUT" && !["checkbox", "radio", "range", "button", "submit"].includes(e.target.type))
            );
            if (isTextEntry || isModalOpen) {
                return;
            }
            // A focused dropdown (e.g. camera selector) must not swallow keys 1-4
            if (e.target && e.target.tagName === "SELECT") {
                e.target.blur();
            }

            if (e.key === "c" || e.key === "C") {
                this.triggerRecenter();
                return;
            }

            // Map keyboard numbers 1-4 for the four patient cards:
            // 1: Food (Top-Left)
            // 2: Water (Top-Right)
            // 3: Medicine (Bottom-Left)
            // 4: Help (Bottom-Right)
            const keyBlockMap = {
                "1": "food", "Numpad1": "food",
                "2": "water", "Numpad2": "water",
                "3": "medicine", "Numpad3": "medicine",
                "4": "help", "Numpad4": "help"
            };
            const targetBlockId = keyBlockMap[e.key] || keyBlockMap[e.code];
            if (targetBlockId) {
                // If user presses numbers 1-4 while calibration is open (e.g. camera not working),
                // automatically skip calibration and dispatch the emergency request immediately!
                if (this.isCalibrating) {
                    this.skipCalibration();
                }
                if (!this.isLocked) {
                    e.preventDefault();
                    this.executeTrigger(targetBlockId, "keyboard");
                }
            }
        });

        // Toggle Camera PIP HUD
        document.getElementById("btn-toggle-cam").addEventListener("click", () => {
            this.pipWindow.classList.toggle("hidden");
        });
        document.getElementById("btn-close-pip").addEventListener("click", () => {
            this.pipWindow.classList.add("hidden");
        });

        // Draggable PiP Window
        const pipHeader = this.pipWindow ? (this.pipWindow.querySelector(".cam-pip-header") || this.pipWindow.querySelector(".pip-header")) : null;
        if (this.pipWindow && pipHeader) {
            let isDragging = false;
            let startX = 0, startY = 0, initialLeft = 0, initialTop = 0;

            const onMouseDown = (e) => {
                if (e.target.closest("button") || e.target.closest("a") || e.target.closest("input")) return;
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                const rect = this.pipWindow.getBoundingClientRect();
                initialLeft = rect.left;
                initialTop = rect.top;
                this.pipWindow.style.bottom = "auto";
                this.pipWindow.style.right = "auto";
                this.pipWindow.style.left = `${initialLeft}px`;
                this.pipWindow.style.top = `${initialTop}px`;
                this.pipWindow.style.transition = "none";
                document.addEventListener("mousemove", onMouseMove);
                document.addEventListener("mouseup", onMouseUp);
                e.preventDefault();
            };

            const onMouseMove = (e) => {
                if (!isDragging) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                const maxLeft = Math.max(0, window.innerWidth - this.pipWindow.offsetWidth);
                const maxTop = Math.max(0, window.innerHeight - this.pipWindow.offsetHeight);
                const newLeft = Math.max(0, Math.min(maxLeft, initialLeft + dx));
                const newTop = Math.max(0, Math.min(maxTop, initialTop + dy));
                this.pipWindow.style.left = `${newLeft}px`;
                this.pipWindow.style.top = `${newTop}px`;
            };

            const onMouseUp = () => {
                if (!isDragging) return;
                isDragging = false;
                this.pipWindow.style.transition = "";
                document.removeEventListener("mousemove", onMouseMove);
                document.removeEventListener("mouseup", onMouseUp);
            };

            pipHeader.addEventListener("mousedown", onMouseDown);
        }

        // Manual Caregiver Override Unlock Button
        document.getElementById("btn-manual-unlock").addEventListener("click", () => {
            this.forceUnlock();
        });

        // Modals: Logs and Settings
        document.getElementById("btn-open-logs").addEventListener("click", () => this.openLogs());
        document.getElementById("btn-close-logs").addEventListener("click", () => this.logsModal.classList.add("hidden"));
        document.getElementById("btn-clear-logs").addEventListener("click", () => this.clearLogs());

        document.getElementById("btn-open-settings").addEventListener("click", () => this.openSettings());
        document.getElementById("btn-close-settings").addEventListener("click", () => this.settingsModal.classList.add("hidden"));
        document.getElementById("btn-cancel-settings").addEventListener("click", () => this.settingsModal.classList.add("hidden"));

        // Settings Range Sliders Live Value Labels
        const dwellSlider = document.getElementById("cfg-dwell-time");
        if (dwellSlider) {
            dwellSlider.addEventListener("input", (e) => {
                document.getElementById("val-dwell-time").textContent = `${e.target.value}s`;
            });
        }
        const calibSlider = document.getElementById("cfg-calib-dwell-time");
        if (calibSlider) {
            calibSlider.addEventListener("input", (e) => {
                const label = document.getElementById("val-calib-dwell-time");
                if (label) label.textContent = `${e.target.value}s`;
            });
        }
        const headSensSlider = document.getElementById("cfg-head-sensitivity");
        if (headSensSlider) {
            headSensSlider.addEventListener("input", (e) => {
                document.getElementById("val-head-sens").textContent = e.target.value;
            });
        }
        const eyeSensSlider = document.getElementById("cfg-eye-sensitivity");
        if (eyeSensSlider) {
            eyeSensSlider.addEventListener("input", (e) => {
                document.getElementById("val-eye-sens").textContent = e.target.value;
            });
        }

        // Settings Form Submit
        const settingsForm = document.getElementById("settings-form");
        if (settingsForm) {
            settingsForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.saveSettings();
            });
        }

        // Test Email Button
        const testEmailBtn = document.getElementById("btn-test-email");
        if (testEmailBtn) {
            testEmailBtn.addEventListener("click", () => this.sendTestEmail());
        }

        // Show/Hide Brevo API Key Toggle in Settings Modal
        const btnToggleModalKey = document.getElementById("btn-toggle-modal-brevo-key");
        const modalKeyInput = document.getElementById("cfg-brevo-api-key");
        const modalKeyIcon = document.getElementById("icon-modal-brevo-key");
        if (btnToggleModalKey && modalKeyInput && modalKeyIcon) {
            btnToggleModalKey.addEventListener("click", () => {
                if (modalKeyInput.type === "password") {
                    modalKeyInput.type = "text";
                    modalKeyIcon.className = "bi bi-eye-slash";
                } else {
                    modalKeyInput.type = "password";
                    modalKeyIcon.className = "bi bi-eye";
                }
            });
        }

        // Simulation Mode Checkbox Dynamic Badge & Text Listener
        const simCheckbox = document.getElementById("cfg-simulation-mode");
        if (simCheckbox) {
            simCheckbox.addEventListener("change", (e) => {
                this.updateSimulationModeUI(e.target.checked);
            });
        }

        // Click on request card also triggers immediately
        this.cards.forEach(card => {
            card.addEventListener("click", () => {
                if (!this.isLocked && !this.isCalibrating) {
                    const blockId = card.getAttribute("data-block-id");
                    this.executeTrigger(blockId, "click");
                }
            });
        });
    }

    /**
     * Toggle Camera On or Off
     */
    async toggleCameraPower() {
        try {
            const nextAction = this.cameraRunning ? "stop" : "start";
            const res = await fetch("/api/camera/toggle", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({ action: nextAction })
            });
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();
            this.cameraRunning = Boolean(data.is_running);
            this.updateCameraUI();
            this.showToast(data.message || (this.cameraRunning ? "Camera turned on." : "Camera turned off."));
        } catch (e) {
            console.error("Camera toggle failed:", e);
            this.showToast("Could not communicate with camera service.");
        }
    }

    updateCameraUI() {
        if (!this.camPowerBtn) return;
        if (this.cameraRunning) {
            if (this.camPowerIcon) this.camPowerIcon.className = "bi bi-camera-video-off";
            if (this.camPowerText) this.camPowerText.textContent = "Turn Off Camera";
            this.camPowerBtn.classList.remove("danger", "btn-danger");
            if (this.visionStatusText) this.visionStatusText.textContent = "Both-Eye Tracking Active";
            if (this.cursorEl && this.hasReceivedValidGaze && this.faceDetected && !this.isLocked) {
                this.cursorEl.style.display = "flex";
                this.cursorEl.style.opacity = "1";
            }
            document.body.classList.remove("face-lost-active");
        } else {
            if (this.camPowerIcon) this.camPowerIcon.className = "bi bi-camera-video";
            if (this.camPowerText) this.camPowerText.textContent = "Turn On Camera";
            this.camPowerBtn.classList.add("danger", "btn-danger");
            if (this.visionStatusText) this.visionStatusText.textContent = "Camera Off — Keys 1-4 Active";
            if (this.statusDot) {
                this.statusDot.classList.remove("live");
                this.statusDot.classList.remove("face-lost");
            }
            if (this.cursorEl) {
                this.cursorEl.style.display = "none";
                this.cursorEl.style.opacity = "0";
            }
            if (this.faceLostBanner) this.faceLostBanner.classList.remove("visible");
            document.body.classList.remove("face-lost-active");
        }
    }

    /**
     * Probe system cameras and populate quick dropdown and settings modal
     */
    async fetchCameras() {
        try {
            const res = await fetch("/api/cameras");
            const data = await res.json();
            if (data.success && data.cameras) {
                this.availableCameras = data.cameras;
                this.cameraIndex = (data.active_index !== undefined) ? data.active_index : 0;
                this.mirrorView = (data.mirror_view !== undefined) ? data.mirror_view : true;

                this.populateCameraOptions();
                this.updateMirrorUI();
            }
        } catch (e) {
            console.warn("Could not query camera devices", e);
        }
    }

    populateCameraOptions() {
        const populateSelect = (selectEl) => {
            if (!selectEl) return;
            selectEl.innerHTML = "";
            if (!this.availableCameras || this.availableCameras.length === 0) {
                const optSys = document.createElement("option");
                optSys.value = "0";
                optSys.textContent = "System Camera (Built-in)";
                optSys.selected = (this.cameraIndex === 0);
                selectEl.appendChild(optSys);

                const optExt = document.createElement("option");
                optExt.value = "1";
                optExt.textContent = "External Camera (1)";
                optExt.selected = (this.cameraIndex === 1);
                selectEl.appendChild(optExt);
                return;
            }

            this.availableCameras.forEach(cam => {
                const opt = document.createElement("option");
                opt.value = cam.index;
                opt.textContent = cam.label || (cam.is_external ? `External Camera (${cam.index})` : `System Camera (${cam.index})`);
                opt.selected = (cam.index === this.cameraIndex);
                selectEl.appendChild(opt);
            });
        };

        populateSelect(this.quickCamSelect);
        populateSelect(this.cfgCamSelect);

        if (this.quickCamSelect) this.quickCamSelect.value = String(this.cameraIndex);
        if (this.cfgCamSelect) this.cfgCamSelect.value = String(this.cameraIndex);
        if (this.pipCamLabel) {
            const current = (this.availableCameras || []).find(c => c.index === this.cameraIndex);
            this.pipCamLabel.textContent = current ? (current.is_external ? "External Cam" : "System Cam") : `Cam ${this.cameraIndex}`;
        }
    }

    async switchCamera(newIndex) {
        newIndex = parseInt(newIndex, 10);
        try {
            this.showToast(`Connecting to Camera ${newIndex}...`);
            const res = await fetch("/api/camera/select", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({ camera_index: newIndex })
            });

            const data = await res.json();
            if (data.success) {
                const activeIdx = (data.camera_index !== undefined) ? data.camera_index : newIndex;
                this.cameraIndex = activeIdx;
                if (this.quickCamSelect) this.quickCamSelect.value = String(activeIdx);
                if (this.cfgCamSelect) this.cfgCamSelect.value = String(activeIdx);
                if (this.pipCamLabel) {
                    const isExt = activeIdx >= 1;
                    this.pipCamLabel.textContent = isExt ? `Ext Cam (${activeIdx})` : `System Cam (${activeIdx})`;
                }
                this.showToast(data.message || `Switched to Camera ${activeIdx}`);
                // Refresh MJPEG stream
                if (this.pipVideoStream) {
                    const baseSrc = this.pipVideoStream.src.split("?")[0];
                    this.pipVideoStream.src = `${baseSrc}?t=${Date.now()}`;
                }
            } else {
                this.showToast(data.error || "Could not switch camera.");
            }
        } catch (e) {
            console.error("Camera switch error", e);
            this.showToast("Network error switching camera.");
        }
    }

    async toggleMirror(desiredState = null) {
        try {
            const nextState = (desiredState !== null) ? desiredState : !this.mirrorView;
            const res = await fetch("/api/camera/mirror", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({ mirror: nextState })
            });

            const data = await res.json();
            if (data.success) {
                this.mirrorView = Boolean(data.mirror_view);
                this.updateMirrorUI();
                this.showToast(data.message || `Mirror view ${this.mirrorView ? "Enabled" : "Disabled"}`);
            }
        } catch (e) {
            this.mirrorView = !this.mirrorView;
            this.updateMirrorUI();
        }
    }

    updateMirrorUI() {
        if (this.mirrorBtnText) {
            this.mirrorBtnText.textContent = this.mirrorView ? "Mirrored" : "Standard";
        }
        if (this.btnToggleMirror) {
            if (this.mirrorView) {
                this.btnToggleMirror.classList.add("primary");
            } else {
                this.btnToggleMirror.classList.remove("primary");
            }
        }
        const cfgMirrorSelect = document.getElementById("cfg-mirror-view");
        if (cfgMirrorSelect) {
            cfgMirrorSelect.value = this.mirrorView ? "1" : "0";
        }
        const pipTitle = document.getElementById("pip-title-text");
        if (pipTitle) {
            pipTitle.textContent = this.mirrorView ? "AI Vision HUD (Mirrored View)" : "AI Vision HUD (Standard View)";
        }
    }

    /**
     * =========================================================================
     * 5-BOX INTERACTIVE EYE CALIBRATION (SINGLE EYE)
     * - Live high-visibility glowing Iris Pointer visible during calibration.
     * - Mandatory: User must fixate inside all 5 boxes to unlock cards.
     * - No skipping or timeout bypassing until calibration is completed.
     * - Computes accuracy and verifies >= 85%.
     * - Dispatches automated confirmation mail to caregiver.
     * =========================================================================
     */
    async startCalibration() {
        if (this.isLocked) return;



        this.isCalibrating = true;
        if (this.pipWindow) this.pipWindow.classList.remove("hidden");
        this.currentCalibIndex = 0;
        this.calibInsideTime = 0;
        this.acceptedSamplesInBox = 0;
        this.lastCalibFrameTime = Date.now();
        this._calibStaleShown = false;
        this._calibExpiredShown = false;

        if (this.calibBoxTimeout) {
            clearTimeout(this.calibBoxTimeout);
            this.calibBoxTimeout = null;
        }

        // Show calibration modal
        if (this.calibrationModal) {
            this.calibrationModal.classList.remove("hidden");
        }

        // Activate high-visibility IRIS POINTER reticle
        if (this.cursorEl) {
            this.cursorEl.classList.add("calibrating");
            this.cursorEl.setAttribute("data-label-hidden", "true");
            this.cursorEl.style.opacity = "1";
            this.cursorEl.style.pointerEvents = "none";
        }

        // Lock cards while calibration is underway
        if (this.blocksGrid) {
            this.blocksGrid.classList.add("cards-locked-for-calibration");
        }

        // Reset all 5 target cards to WAITING
        this.calibrationTargets = document.querySelectorAll(".calibration-target");
        this.calibrationTargets.forEach(target => {
            target.classList.remove("active-target", "gaze-inside", "completed-target");
            const statusEl = target.querySelector(".target-status");
            if (statusEl) {
                statusEl.textContent = "WAITING";
                statusEl.className = "target-status";
            }
        });

        try {
            const res = await fetch("/api/calibration/start", { method: "POST", headers: this.getHeaders() });

        } catch (err) {
            console.warn("[Calibration] Start error:", err);
            if (this.cursorEl) {
                this.cursorEl.classList.remove("calibrating");
                this.cursorEl.removeAttribute("data-label-hidden");
            }
            this.isCalibrating = false;
            if (this.calibrationModal) this.calibrationModal.classList.add("hidden");
            if (this.blocksGrid) this.blocksGrid.classList.remove("cards-locked-for-calibration");
            return;
        }

        this.showToast("Both-Eye Calibration: Steer the Iris Pointer inside each highlighted target");
        this.runCalibrationBox(0);
    }

    async runCalibrationBox(index) {
        if (!this.isCalibrating) return;

        if (index >= this.calibBoxes.length) {
            await this.finishCalibration();
            return;
        }

        this.currentCalibIndex = index;
        const box = this.calibBoxes[index];

        this.calibInsideTime = 0;
        this.calibDwellActive = false;
        this.calibDwellStartTime = null;
        this.calibBlinkGraceStartTime = null;
        this.acceptedSamplesInBox = 0;
        this.lastSampleSentTime = 0;
        this.lastCalibFrameTime = Date.now();

        if (!this.calibrationTargets || this.calibrationTargets.length === 0) {
            this.calibrationTargets = document.querySelectorAll(".calibration-target");
        }

        this.calibrationTargets.forEach(target => {
            target.classList.remove("active-target", "gaze-inside", "gaze-hover");
            const tid = target.getAttribute("data-target-id");
            if (!target.classList.contains("completed-target")) {
                const statusEl = target.querySelector(".target-status");
                if (statusEl) {
                    statusEl.textContent = "WAITING";
                    statusEl.className = "target-status";
                }
                const circle = document.getElementById(`calib-dwell-circle-${tid}`);
                if (circle) circle.style.strokeDashoffset = this.maxDwellOffset;
                const textEl = document.getElementById(`calib-dwell-text-${tid}`);
                if (textEl) textEl.textContent = "LOOK";
            }
        });

        const activeTarget = document.querySelector(
            `.calibration-target[data-target-id="${box.id}"]`
        );

        if (!activeTarget) {
            console.error("Calibration target missing:", box.id);
            return;
        }

        activeTarget.classList.add("active-target");
        const activeStatus = activeTarget.querySelector(".target-status");
        if (activeStatus) {
            activeStatus.textContent = "ACTIVE";
            activeStatus.className = "target-status badge text-bg-primary";
        }

        const activeCircle = document.getElementById(`calib-dwell-circle-${box.id}`);
        if (activeCircle) activeCircle.style.strokeDashoffset = this.maxDwellOffset;
        const activeText = document.getElementById(`calib-dwell-text-${box.id}`);
        if (activeText) activeText.textContent = "LOOK";

        this.calibrationBox = activeTarget;

        try {
            await fetch("/api/calibration/box_activate", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({ box_id: box.id, reset: true })
            });
        } catch (err) {
            console.warn("[Calibration] box_activate error:", err);
        }

        const stepBadge = document.getElementById("calib-step-badge");
        if (stepBadge) stepBadge.textContent = `Point ${index + 1} of 5`;

        const pointName = document.getElementById("calib-point-name");
        if (pointName) pointName.textContent = box.name;

        const progBar = document.getElementById("calib-progress-bar");
        if (progBar) progBar.style.width = `${(index / 5) * 100}%`;

        const countdownText = document.getElementById("calib-countdown-text");
        if (countdownText) countdownText.textContent = "Look at and hold your cursor inside the highlighted target";

        if (this.calibInstruction) this.calibInstruction.textContent = "Look at and hold your cursor inside the highlighted target";
        if (this.calibBoxBadge) this.calibBoxBadge.textContent = `POINT ${index + 1}: ${box.name.toUpperCase()}`;
        if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Look at and hold your cursor inside the highlighted target";

        this.playTone(580 + index * 55, "sine", 0.12, 0.08);

        if (this.calibBoxTimeout) {
            clearTimeout(this.calibBoxTimeout);
            this.calibBoxTimeout = null;
        }
    }

    /**
     * Card-style gaze dwell for calibration target activation.
     * Enforces visible cursor hit-testing against highlighted target rectangle,
     * fresh frame, strict both-eye tracking (no cursor-only single-eye fallback),
     * brief micro-blink pause (<=350ms), immediate reset on cursor exit / prolonged blink / face lost / stale frame,
     * sample collection stream every 75ms, and completion only when BOTH dwell >= duration AND samples >= 16.
     */
    async updateCalibrationBoxDwell() {
        if (!this.isCalibrating || !this.calibrationBox) return;

        const box = this.calibBoxes[this.currentCalibIndex];
        if (!box) return;

        const now = Date.now();
        const TARGET_SAMPLES = 12;
        const dwellDuration = this.calibrationDwellDuration || 1500;
        const isFresh = (now - this.lastTelemetryReceiveTime) <= 400;

        // Strict calibration quality gates:
        // Must require fresh valid both-eye tracking, no blink, valid face detection.
        // Single-eye cursor fallback MUST NOT advance calibration or collect samples.
        const isGazeStrictlyValid = (
            this.cameraRunning &&
            isFresh &&
            this.faceDetected &&
            this.gazeValid &&
            this.bothEyesValid &&
            !this.isBlinking &&
            !this.isStalePrediction
        );

        // Screen-coordinate hit testing against active target element rectangle (exact card-dwell approach)
        const screenX = this.displayGazeX * window.innerWidth;
        const screenY = this.displayGazeY * window.innerHeight;
        const rect = this.calibrationBox.getBoundingClientRect();
        const margin = this.calibDwellActive ? 16 : 0;
        const isCursorInside = (
            screenX >= rect.left - margin &&
            screenX <= rect.right + margin &&
            screenY >= rect.top - margin &&
            screenY <= rect.bottom + margin
        );

        // Micro-blink grace handling (up to 350ms):
        // If cursor is physically inside the target and tracking was active, but temporary blink occurs:
        if (isCursorInside && this.cameraRunning && isFresh && this.faceDetected && this.isBlinking) {
            if (!this.calibBlinkGraceStartTime) {
                this.calibBlinkGraceStartTime = now;
            }
            const blinkDuration = now - this.calibBlinkGraceStartTime;
            if (blinkDuration <= 350) {
                // Pause dwell progression during micro-blink
                if (this.lastCalibFrameTime && this.calibDwellStartTime) {
                    this.calibDwellStartTime += (now - this.lastCalibFrameTime);
                }
                this.lastCalibFrameTime = now;
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "BLINK DETECTED — PAUSED";
                return;
            }
        }

        // If tracking drops, prolonged blink (>350ms), face loss, stale frame, or cursor exits target:
        if (!isGazeStrictlyValid || !isCursorInside) {
            this.calibBlinkGraceStartTime = null;
            this.lastCalibFrameTime = now;

            const hadProgress = this.calibDwellActive || (this.calibDwellStartTime !== null) || (this.acceptedSamplesInBox > 0);

            this.calibDwellActive = false;
            this.calibDwellStartTime = null;
            // Retain samples to allow brief exit recovery
            // this.acceptedSamplesInBox = 0;

            // Reset visual indicators on active target
            this.calibrationBox.classList.remove("gaze-inside", "gaze-hover");
            const circle = document.getElementById(`calib-dwell-circle-${box.id}`);
            if (circle) circle.style.strokeDashoffset = this.maxDwellOffset;
            const textEl = document.getElementById(`calib-dwell-text-${box.id}`);
            if (textEl) textEl.textContent = "LOOK";

            const statusEl = this.calibrationBox.querySelector(".target-status");
            if (statusEl && !this.calibrationBox.classList.contains("completed-target")) {
                statusEl.textContent = "ACTIVE";
                statusEl.className = "target-status badge text-bg-primary";
            }

            const basePct = (this.currentCalibIndex / 5) * 100;
            if (this.calibProgressBar) this.calibProgressBar.style.width = `${basePct}%`;

            // Do not aggressively reset server samples to allow jitter recovery
            // if (hadProgress) {
            //     this.resetActiveBoxSamples(box.id);
            // }

            // Update UI feedback badges
            if (!this.cameraRunning) {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "CAMERA NOT TRACKING";
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Camera is off.";
            } else if (!this.faceDetected) {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "FACE NOT DETECTED";
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Position face in front of camera";
            } else if (this.isBlinking) {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "BLINK DETECTED — RESET";
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Hold cursor inside target with eyes open";
            } else if (!this.bothEyesValid || !this.gazeValid) {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "BOTH EYES REQUIRED";
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Both-eye tracking required for calibration";
            } else if (!isFresh || this.isStalePrediction) {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = "STREAM STALE — WAITING";
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Awaiting fresh video frame";
            } else {
                if (this.calibBoxBadge) this.calibBoxBadge.textContent = `POINT ${this.currentCalibIndex + 1}: ${box.name.toUpperCase()}`;
                if (this.calibBoxSublabel) this.calibBoxSublabel.textContent = "Look at and hold your cursor inside the highlighted target";
                if (this.calibCountdownText) this.calibCountdownText.textContent = "Look at and hold your cursor inside the highlighted target";
            }
            return;
        }

        // Active Dwell inside target
        this.calibBlinkGraceStartTime = null;
        this.lastCalibFrameTime = now;

        if (!this.calibDwellActive) {
            this.calibDwellActive = true;
            this.calibDwellStartTime = now;
            this.playTone(480, "sine", 0.08, 0.05);
        }

        this.calibrationBox.classList.add("gaze-inside", "gaze-hover");

        const dwellElapsed = now - this.calibDwellStartTime;
        const dwellProgress = Math.min(1.0, dwellElapsed / dwellDuration);

        // Update radial SVG circle & text matching communication card dwell
        const circle = document.getElementById(`calib-dwell-circle-${box.id}`);
        if (circle) {
            circle.style.strokeDashoffset = this.maxDwellOffset * (1.0 - dwellProgress);
        }
        const textEl = document.getElementById(`calib-dwell-text-${box.id}`);
        const remainingSec = Math.max(0, (dwellDuration - dwellElapsed) / 1000).toFixed(1);
        if (textEl) {
            textEl.textContent = `${remainingSec}s`;
        }

        if (this.calibBoxBadge) {
            this.calibBoxBadge.textContent = "HOLD CURSOR INSIDE TARGET";
        }
        if (this.calibBoxSublabel) {
            this.calibBoxSublabel.textContent = `Collecting samples (${this.acceptedSamplesInBox}/${TARGET_SAMPLES}) • ${remainingSec}s remaining`;
        }
        if (this.calibCountdownText) {
            this.calibCountdownText.textContent = `Hold cursor inside target: ${remainingSec}s`;
        }

        const statusEl = this.calibrationBox.querySelector(".target-status");
        if (statusEl) {
            statusEl.textContent = `DWELLING (${this.acceptedSamplesInBox}/${TARGET_SAMPLES})`;
            statusEl.className = "target-status badge text-bg-warning";
        }

        // Update top progress bar across all 5 targets
        const basePct = (this.currentCalibIndex / 5) * 100;
        const dwellFrac = Math.min(1.0, dwellElapsed / dwellDuration);
        const sampleFrac = Math.min(1.0, this.acceptedSamplesInBox / TARGET_SAMPLES);
        const combinedFrac = Math.min(1.0, (dwellFrac + sampleFrac) / 2.0);
        const totalPct = Math.min(100, Math.round(basePct + (combinedFrac * 20)));
        if (this.calibProgressBar) {
            this.calibProgressBar.style.width = `${totalPct}%`;
        }

        // Stream raw gaze samples to backend every 75ms only while gaze is strictly valid and cursor is inside
        if (now - this.lastSampleSentTime >= 75 && !this.isSendingSample) {
            this.lastSampleSentTime = now;
            this.isSendingSample = true;
            const sampleX = this.lastRawGazeX !== undefined ? this.lastRawGazeX : this.displayGazeX;
            const sampleY = this.lastRawGazeY !== undefined ? this.lastRawGazeY : this.displayGazeY;
            try {
                const res = await fetch("/api/calibration/sample", {
                    method: "POST",
                    headers: this.getHeaders(),
                    body: JSON.stringify({
                        box_id: box.id,
                        raw_x: sampleX,
                        raw_y: sampleY
                    })
                });
                const data = await res.json();
                if (data.accepted) {
                    this.acceptedSamplesInBox = data.sample_count || (this.acceptedSamplesInBox + 1);
                    if (this.acceptedSamplesInBox % 4 === 0) {
                        this.playTone(660, "sine", 0.04, 0.03);
                    }
                }
            } catch (err) {
                console.warn("[Calibration] Sample error:", err);
            } finally {
                this.isSendingSample = false;
            }
        }

        // Complete point ONLY when continuous dwell duration is reached AND at least 12 server-accepted samples
        if (dwellElapsed >= dwellDuration || this.acceptedSamplesInBox >= TARGET_SAMPLES) {
            this.calibDwellActive = false;
            this.calibDwellStartTime = null;
            this.calibrationBox.classList.remove("active-target", "gaze-inside", "gaze-hover");
            this.calibrationBox.classList.add("completed-target");
            if (circle) circle.style.strokeDashoffset = 0;
            if (textEl) textEl.textContent = "DONE";
            if (statusEl) {
                statusEl.textContent = "COMPLETED";
                statusEl.className = "target-status badge text-bg-success";
            }
            this.playChimeStep();
            this.runCalibrationBox(this.currentCalibIndex + 1);
        }
    }

    async resetActiveBoxSamples(boxId) {
        try {
            await fetch("/api/calibration/box_activate", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({ box_id: boxId, reset: true })
            });
        } catch (e) { }
    }

    async finishCalibration() {
        if (this.calibBoxTimeout) clearTimeout(this.calibBoxTimeout);
        this.isCalibrating = false;

        if (this.calibProgressBar) {
            this.calibProgressBar.style.width = "100%";
        }

        let result = { success: false, calibrated: false, accuracy: null, quality: "None", message: "Calibration finalizing..." };
        try {
            const res = await fetch("/api/calibration/finish", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({
                    screen_w: window.innerWidth || 1920,
                    screen_h: window.innerHeight || 1080
                })
            });
            result = await res.json();
        } catch (e) {
            console.error("Error finalizing calibration", e);
        }

        if (result.success && result.calibrated) {
            this.isCalibrated = true;
            try {
                fetch("/api/calibration/notify", { method: "POST", headers: this.getHeaders() }).catch(() => { });
            } catch (e) { }

            const accText = (result.accuracy !== undefined && result.accuracy !== null) ? `${result.accuracy}%` : "Not Measured";
            const errDetail = result.val_mae_px ? ` | Val MAE: ${result.val_mae_px}px` : "";
            this.playChimeDelivered();
            this.showToast(`Both-Eye Calibration: ${result.quality || "Processed"} (Hit Rate: ${accText}${errDetail})`);
        } else {
            this.isCalibrated = false;
            this.showToast("Calibration quality too low. Tips: improve lighting, reduce head movement, clean the camera lens, then click Re-Calibrate to try again.");
        }

        if (this.cursorEl) {
            this.cursorEl.classList.remove("calibrating");
            this.cursorEl.removeAttribute("data-label-hidden");
        }

        if (this.calibrationModal) {
            this.calibrationModal.classList.add("hidden");
        }

        if (this.blocksGrid) {
            this.blocksGrid.classList.remove("cards-locked-for-calibration");
        }

        this.fetchCameras();
    }

    /**
     * Skip calibration: immediately unlocks the communication cards and activates
     * keyboard emergency hotkeys 1-4 so emergency emails can be dispatched.
     */
    async skipCalibration() {
        try {
            const resp = await fetch("/api/calibration/skip", { method: "POST", headers: this.getHeaders() });
            if (resp.status === 401) {
                this.showToast("Session expired — please log in again.");
                return;
            }
        } catch (e) {
            console.warn("[Calibration] Skip error:", e);
        }

        if (this.calibBoxTimeout) clearTimeout(this.calibBoxTimeout);
        this.isCalibrating = false;

        if (this.calibrationModal) {
            this.calibrationModal.classList.add("hidden");
        }

        if (this.cursorEl) {
            this.cursorEl.classList.remove("calibrating");
            this.cursorEl.removeAttribute("data-label-hidden");
        }

        this.isCalibrated = false;
        if (this.blocksGrid) {
            this.blocksGrid.classList.remove("cards-locked-for-calibration");
        }

        this.showToast("Calibration Skipped: Adaptive resting baseline active (Keys 1-4 ready)");
        this.fetchCameras();
    }

    /**
     * Connect to Server-Sent Events (SSE) for low-latency live telemetry.
     */
    startSSE() {
        const evtSource = new EventSource("/api/stream");

        evtSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleTelemetry(data);
            } catch (err) {
                console.error("SSE parse error", err);
            }
        };

        evtSource.onerror = (err) => {
            this.statusDot.classList.remove("live");
            this.visionStatusText.textContent = "Reconnecting Stream...";
        };
    }

    /**
     * Process live telemetry frame from vision engine
     */
    handleTelemetry(data) {
        if (!data) return;

        this.lastTelemetryReceiveTime = Date.now();
        this.lastFrameTimestamp = data.frame_timestamp || Date.now();
        this.isStalePrediction = Boolean(data.stale_prediction);

        // Always synchronize server lockout state & alert status first, even if camera is inactive
        if (data.lock) {
            this.syncLockState(data.lock);
        }



        // Check if camera is disabled or unavailable on backend
        const camActive = (data.camera_enabled !== false) && (data.camera_available !== false);
        if (!camActive) {
            this.cameraRunning = false;
            this.updateCameraUI();
            this.resetAllBlockDwells();
            this.selectionState = 'CAMERA_UNAVAILABLE';
            if (this.visionStatusText && !this.isLocked) {
                this.visionStatusText.textContent = "Camera Off — Emergency Keys 1-4 Active";
            }
            return;
        } else if (!this.cameraRunning) {
            this.cameraRunning = true;
            this.updateCameraUI();
            this.selectionState = 'TRACKING';
        }

        this.faceDetected = Boolean(data.face_detected);
        if (this.faceDetected) {
            this.hasEverTrackedFace = true;
        }
        this.gazeValid = (data.gaze_valid !== undefined) ? Boolean(data.gaze_valid) : true;
        this.bothEyesValid = (data.both_eyes_valid !== undefined) ? Boolean(data.both_eyes_valid) : true;
        this.isBlinking = Boolean(data.is_blinking);
        this.usableForSelection = (data.usable_for_selection !== undefined) ? Boolean(data.usable_for_selection) : true;
        if (data.calibrated !== undefined && data.calibrated) {
            this.isCalibrated = true;
        }
        this.cursorValid = (data.cursor_valid !== undefined) ? Boolean(data.cursor_valid) : (this.gazeValid && this.bothEyesValid);

        // Handle calibration stale notification (CP-3)
        if (data.calibration_stale && !this._calibStaleNotified && !this.isCalibrating) {
            this._calibStaleNotified = true;
            this.showToast("⚠️ Calibration reset — please re-calibrate for optimal accuracy.");
        } else if (!data.calibration_stale) {
            this._calibStaleNotified = false;
        }

        // Handle calibration expired notification (CP-4)
        if (data.calibration_expired && !this._calibExpiredNotified) {
            this._calibExpiredNotified = true;
            this.showToast("⚠️ Saved calibration profile expired (>24h). Recalibration recommended.");
        }

        // Real-time calibration rejection reason from backend (CP-2)
        if (this.isCalibrating && data.calibration_reject_reason && this.calibBoxSublabel) {
            this.calibBoxSublabel.textContent = data.calibration_reject_reason;
        }

        // Restore dynamic cursor movement from fresh, valid predictions only.
        // Do NOT overwrite last valid coordinate with fabricated or stale coordinates!
        if (this.gazeValid && this.bothEyesValid && !this.isStalePrediction && data.gaze_x !== undefined) {
            this.lastRawGazeX = data.raw_gaze_x !== undefined ? data.raw_gaze_x : data.gaze_x;
            this.lastRawGazeY = data.raw_gaze_y !== undefined ? data.raw_gaze_y : data.gaze_y;
        }
        if (this.cursorValid && data.cursor_x !== undefined) {
            this.lastGazeX = data.cursor_x;
            this.lastGazeY = data.cursor_y;
            if (!this.hasReceivedValidGaze) {
                this.displayGazeX = data.cursor_x;
                this.displayGazeY = data.cursor_y;
                this.hasReceivedValidGaze = true;
            }
        }

        // Update Head Pose status
        this.headPoseText.textContent = `Yaw: ${data.head_yaw}° | Pitch: ${data.head_pitch}°`;

        // Update Status indicator
        if (this.faceDetected) {
            this.statusDot.classList.add("live");
            this.statusDot.classList.remove("face-lost");
            if (this.isStalePrediction) {
                this.visionStatusText.textContent = "Camera Frame Stale / Frozen — Dwell Paused";
            } else {
                this.visionStatusText.textContent = (!data.gaze_valid && data.invalid_reason) ? `Tracking: ${data.invalid_reason}` : (data.status || "Both-Eye Tracking Active");
            }
            if (this.faceLostBanner) {
                this.faceLostBanner.classList.remove("visible");
                this.faceLostBanner.classList.add("d-none");
            }
            document.body.classList.remove("face-lost-active");
        } else {
            this.statusDot.classList.remove("live");
            this.statusDot.classList.add("face-lost");
            this.visionStatusText.textContent = "No Face Detected — Dwell Paused (Keys 1-4 Ready)";
            if (this.faceLostBanner && !this.isCalibrating && this.cameraRunning) {
                this.faceLostBanner.classList.add("visible");
                this.faceLostBanner.classList.remove("d-none");
            }
            document.body.classList.add("face-lost-active");

            // Cancel ongoing gaze dwell and clear all highlights if face is lost
            this.resetAllBlockDwells();
            this.selectionState = 'IDLE';
        }

        // Update PIP stats
        const pipFps = document.getElementById("pip-fps-text");
        const pipBlink = document.getElementById("pip-blink-text");
        if (pipFps) pipFps.textContent = `${data.fps || 30} FPS`;
        if (pipBlink) {
            pipBlink.textContent = data.is_blinking ? "Blinking" : "Both Eyes Tracked";
            pipBlink.style.color = data.is_blinking ? "#dc2626" : "#10b981";
        }
        if (data.camera_name && this.pipCamLabel) {
            const isExt = data.camera_index >= 1;
            this.pipCamLabel.textContent = isExt ? `Ext Cam (${data.camera_index})` : `System Cam (${data.camera_index})`;
        }
        if (data.camera_index !== undefined && data.camera_index !== this.cameraIndex) {
            this.cameraIndex = data.camera_index;
            if (this.quickCamSelect) this.quickCamSelect.value = String(data.camera_index);
            if (this.cfgCamSelect) this.cfgCamSelect.value = String(data.camera_index);
        }

        // Telemetry loop lock and alert sync
        if (data.lock) {
            this.syncLockState(data.lock);
        }
    }

    /**
     * Real-time 60 FPS Render & Hit-Testing Loop
     * Implements responsive exponential damping (lerp) to move the pointer smoothly without freezing!
     */
    startRenderLoop() {
        const render = () => {
            const now = Date.now();
            const isFresh = (now - this.lastTelemetryReceiveTime) <= 400;
            const trackingValid = (
                this.cameraRunning &&
                this.faceDetected &&
                this.cursorValid &&
                isFresh &&
                this.hasReceivedValidGaze
            );

            // Responsive adaptive interpolation for smooth pointer movement and snappy saccades
            if (trackingValid) {
                const dx = this.lastGazeX - this.displayGazeX;
                const dy = this.lastGazeY - this.displayGazeY;
                const dist = Math.hypot(dx, dy);
                // Dynamic lerp: 0.45 on rapid saccades (> 0.04 normalized dist) for snappy responsiveness,
                // and 0.28 on micro-fixations for stable, jitter-free dwelling.
                const lerpFactor = dist > 0.04 ? 0.45 : 0.28;
                this.displayGazeX += dx * lerpFactor;
                this.displayGazeY += dy * lerpFactor;
            }

            this.updateCursor();

            const isModalOpen = (
                (this.settingsModal && !this.settingsModal.classList.contains("hidden")) ||
                (this.logsModal && !this.logsModal.classList.contains("hidden"))
            );

            if (this.isCalibrating) {
                this.updateCalibrationBoxDwell();
            } else if (!this.isLocked && !isModalOpen) {
                this.hitTestBlocks();
            }
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    /**
     * Update Gaze Cursor position and appearance based on validity, freshness, and tracking health
     */
    updateCursor() {
        if (!this.cursorEl) return;

        // If lockout modal is active or caregiver overlay is present, suppress cursor
        if (this.isLocked || document.body.classList.contains("lockout-active")) {
            this.cursorEl.classList.add("lockout-hidden");
            this.cursorEl.style.display = "none";
            this.cursorEl.style.opacity = "0";
            return;
        } else {
            this.cursorEl.classList.remove("lockout-hidden");
        }

        const now = Date.now();
        const isFresh = (now - this.lastTelemetryReceiveTime) <= 400;
        document.body.classList.toggle("calibrating-active", !!this.isCalibrating);
        const trackingAvailable = (
            this.cameraRunning &&
            this.faceDetected &&
            this.cursorValid &&
            isFresh &&
            this.hasReceivedValidGaze
        );

        if (!trackingAvailable) {
            this.cursorEl.classList.add("tracking-unavailable");
            this.cursorEl.classList.remove("dwelling");

            // When face is not detected, camera is off, or no valid gaze ever received: hide cursor completely
            if (!this.cameraRunning || !this.faceDetected || !this.hasReceivedValidGaze) {
                this.cursorEl.classList.add("no-face");
                this.cursorEl.style.display = "none";
                this.cursorEl.style.opacity = "0";
            } else {
                // Face is detected but eyes temporarily invalid (e.g. natural blink): show subtle dot/ring without crosshair
                this.cursorEl.classList.remove("no-face");
                this.cursorEl.style.display = "flex";
                this.cursorEl.style.opacity = "0.35";
                const screenX = this.displayGazeX * window.innerWidth;
                const screenY = this.displayGazeY * window.innerHeight;
                this.cursorEl.style.transform = `translate(${screenX}px, ${screenY}px) translate(-50%, -50%)`;
            }
            return;
        }

        this.cursorEl.classList.remove("tracking-unavailable", "no-face");
        this.cursorEl.style.display = "flex";
        this.cursorEl.style.opacity = "1";

        const screenX = this.displayGazeX * window.innerWidth;
        const screenY = this.displayGazeY * window.innerHeight;
        this.cursorEl.style.transform = `translate(${screenX}px, ${screenY}px) translate(-50%, -50%)`;

        if (this.isCalibrating) {
            this.cursorEl.classList.add("calibrating");
        } else {
            this.cursorEl.classList.remove("calibrating");
        }

        if (this.activeHoverBlock && !this.isLocked) {
            this.cursorEl.classList.add("dwelling");
        } else {
            this.cursorEl.classList.remove("dwelling");
        }
    }

    /**
     * Hit-test gaze point against the 4 communication patient cards.
     * Enforces the Selection State Machine (IDLE -> TRACKING -> DWELLING -> CONFIRMED -> COOLDOWN -> WAITING_FOR_RELEASE)
     * and Frame Freshness Watchdog with 350ms natural micro-blink tolerance.
     */
    hitTestBlocks() {
        const now = Date.now();



        // 1. Frame Freshness & Hardware Watchdog:
        // Immediate reset upon camera power loss, frozen video frames (> 400ms), face missing, or stale predictions
        const isFresh = (now - this.lastTelemetryReceiveTime) <= 400;
        if (!this.cameraRunning || !isFresh || !this.faceDetected || this.isStalePrediction || !this.hasReceivedValidGaze) {
            this.blinkGraceStartTime = null;
            this.resetAllBlockDwells();
            if (!this.cameraRunning || !isFresh) {
                this.selectionState = 'CAMERA_UNAVAILABLE';
                if (this.visionStatusText && !this.isLocked) {
                    this.visionStatusText.textContent = !this.cameraRunning ? "Camera Turned Off" : "Camera Stream Frozen — Gaze Paused";
                }
            } else if (!this.faceDetected) {
                this.selectionState = 'IDLE';
            } else {
                this.selectionState = 'TRACKING';
            }
            return;
        }

        // 2. Micro-Blink Grace Period (350ms):
        // Face is present and camera is running, but eye tracking is temporarily invalid due to a blink.
        // Natural blinks (100-300ms) pause dwell progress without resetting it.
        // Relaxed selection: Allow single-eye fallback cursor for patient card selection if both-eye drops.
        const isGazeCurrentlyValid = ((this.gazeValid && this.bothEyesValid) || this.cursorValid) && !this.isBlinking;
        if (!isGazeCurrentlyValid) {
            if (this.activeHoverBlock) {
                if (!this.blinkGraceStartTime) {
                    this.blinkGraceStartTime = now;
                }
                const blinkElapsed = now - this.blinkGraceStartTime;
                if (blinkElapsed <= 350) {
                    // Pause dwell time advance so the blink cannot trigger selection or destroy progress
                    if (this.lastFrameTime) {
                        this.dwellStartTime += (now - this.lastFrameTime);
                    }
                    this.lastFrameTime = now;
                    return;
                } else {
                    // Interruption lasted longer than 350ms: prolonged eye closure or looking away
                    this.resetBlockDwell(this.activeHoverBlock);
                    this.activeHoverBlock = null;
                    this.dwellStartTime = null;
                    this.lastDwellTickWindow = -1;
                    this.blinkGraceStartTime = null;
                    this.selectionState = 'TRACKING';
                    this.lastFrameTime = now;
                    return;
                }
            } else {
                this.blinkGraceStartTime = null;
                this.selectionState = 'TRACKING';
                this.lastFrameTime = now;
                return;
            }
        }

        // Tracking is valid and eyes are open: clear blink grace timer
        this.blinkGraceStartTime = null;
        this.lastGazeValidTimestamp = now;
        this.lastFrameTime = now;

        const screenX = this.displayGazeX * window.innerWidth;
        const screenY = this.displayGazeY * window.innerHeight;

        let hoveredBlock = null;

        // Check which of the 4 cards contains the smooth gaze point
        for (const card of this.cards) {
            const rect = card.getBoundingClientRect();
            // Inactive cards require gaze to actually cross into the card boundary (margin 0).
            // Once a card is actively dwelling, expand margin to 16px to prevent jitter from interrupting selection.
            const margin = (this.activeHoverBlock === card.getAttribute("data-block-id")) ? 16 : 0;
            if (
                screenX >= rect.left - margin &&
                screenX <= rect.right + margin &&
                screenY >= rect.top - margin &&
                screenY <= rect.bottom + margin
            ) {
                hoveredBlock = card;
                break;
            }
        }

        // 3. Selection State Machine Handling
        // If WAITING_FOR_RELEASE: verify user has moved gaze away from the previously confirmed card
        if (this.selectionState === 'WAITING_FOR_RELEASE') {
            if (hoveredBlock && hoveredBlock.getAttribute("data-block-id") === this.lastConfirmedCardId) {
                // Patient is still fixating on the same card after confirmation; suppress any new dwell
                const textEl = document.getElementById(`dwell-text-${this.lastConfirmedCardId}`);
                if (textEl) textEl.textContent = "RELEASE";
                return;
            } else {
                // Patient has broken fixation and looked away into neutral area or another card!
                this.lastConfirmedCardId = null;
                this.selectionState = hoveredBlock ? 'DWELLING' : 'TRACKING';
            }
        }

        if (hoveredBlock) {
            const blockId = hoveredBlock.getAttribute("data-block-id");

            if (this.activeHoverBlock !== blockId) {
                if (this.activeHoverBlock) {
                    this.resetBlockDwell(this.activeHoverBlock);
                }
                this.activeHoverBlock = blockId;
                this.dwellStartTime = now;
                this.selectionState = 'DWELLING';
                hoveredBlock.classList.add("gaze-hover");
                this.playTone(480, "sine", 0.08, 0.05);
            }

            // Calculate dwell progress
            const elapsed = now - this.dwellStartTime;
            const progress = Math.min(1.0, elapsed / this.dwellDuration);

            // Update SVG circle stroke-dashoffset (264 down to 0)
            const circle = document.getElementById(`dwell-circle-${blockId}`);
            const textEl = document.getElementById(`dwell-text-${blockId}`);

            if (circle) {
                const offset = this.maxDwellOffset * (1.0 - progress);
                circle.style.strokeDashoffset = offset;
            }

            if (textEl) {
                const remainingSeconds = Math.max(0, (this.dwellDuration - elapsed) / 1000).toFixed(1);
                textEl.textContent = `${remainingSeconds}s`;
            }

            // Periodic audio tick once per 450ms window
            const tickWindow = Math.floor(elapsed / 450);
            if (this.lastDwellTickWindow !== tickWindow && elapsed < this.dwellDuration) {
                this.lastDwellTickWindow = tickWindow;
                this.playTone(550, "sine", 0.04, 0.02);
            }

            // EDGE-TRIGGERED SELECTION: Exactly ONE confirmation per valid fixation!
            if (elapsed >= this.dwellDuration) {
                this.selectionState = 'CONFIRMED';
                this.lastConfirmedCardId = blockId;
                const uniqueEventId = 'evt_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
                this.executeTrigger(blockId, "gaze", uniqueEventId);
            }
        } else {
            // Gaze outside any block: smoothly reset active dwell
            if (this.activeHoverBlock) {
                this.resetBlockDwell(this.activeHoverBlock);
                this.activeHoverBlock = null;
                this.dwellStartTime = null;
                this.lastDwellTickWindow = -1;
            }
            if (this.selectionState !== 'WAITING_FOR_RELEASE') {
                this.selectionState = 'TRACKING';
            }
        }
    }

    /**
     * Reset a block's visual dwell progress
     */
    resetBlockDwell(blockId) {
        const card = document.getElementById(`card-${blockId}`);
        if (card) {
            card.classList.remove("gaze-hover");
        }
        const circle = document.getElementById(`dwell-circle-${blockId}`);
        if (circle) {
            circle.style.strokeDashoffset = this.maxDwellOffset;
        }
        const textEl = document.getElementById(`dwell-text-${blockId}`);
        if (textEl) {
            textEl.textContent = "LOOK";
        }
    }

    resetAllBlockDwells() {
        if (this.cards) {
            this.cards.forEach(c => {
                c.classList.remove("gaze-hover", "active");
                const bId = c.getAttribute("data-block-id");
                if (bId) {
                    this.resetBlockDwell(bId);
                }
            });
        }
        this.activeHoverBlock = null;
        this.dwellStartTime = null;
        this.lastDwellTickWindow = -1;
    }

    /**
     * TRIGGER & STRICT LOCKOUT PROTOCOL
     * Triggered either by sustained eye gaze dwell OR pressing keyboard keys 1 through 4.
     * 1. Engage immediate system lock.
     * 2. Freeze all other cards (no other card can be selected).
     * 3. Send email to caregiver's email with unique idempotency event ID.
     * 4. Keep locked until email is dispatched and confirmed.
     */
    async executeTrigger(blockId, triggerSource = "gaze", eventId = null) {
        if (this.isLocked) return;



        // Freshness and safety check for gaze triggers
        if (triggerSource === "gaze") {
            const now = Date.now();
            const isFresh = (now - this.lastTelemetryReceiveTime) <= 400;
            const gazeRecentlyValid = (now - (this.lastGazeValidTimestamp || 0)) <= 600;
            if (!this.cameraRunning || !isFresh || !this.faceDetected || !gazeRecentlyValid) {
                console.warn("[Trigger] Rejected: camera frozen, face missing, or gaze invalid.");
                this.resetAllBlockDwells();
                this.selectionState = 'CAMERA_UNAVAILABLE';
                return;
            }
        }

        const card = document.getElementById(`card-${blockId}`);
        if (!card) return;

        const title = card.getAttribute("data-title");
        const priority = card.getAttribute("data-priority");
        const message = card.getAttribute("data-message");
        const category = card.getAttribute("data-category");

        if (!eventId) {
            eventId = 'evt_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
        }

        this.lastTriggerSentTime = Date.now();
        this.engageLockUI(blockId, title, priority, message, triggerSource);
        this.playChimeLock();

        try {
            const response = await fetch("/api/trigger_block", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify({
                    block_id: blockId,
                    block_title: title,
                    category: category,
                    priority: priority,
                    message: message,
                    trigger_source: triggerSource,
                    event_id: eventId
                })
            });

            const data = await response.json();
            if (!data.success) {
                console.error("Lockout trigger error", data.error);
                this.releaseLockUI();
                this.showToast(data.error || "Notification dispatch rejected.", "danger");
                this.resetAllBlockDwells();
            } else {
                console.log("Trigger successful, request ID:", data.request_id);
                // Advance Step 3 (Database Logged) to completed state
                const stepDb = document.getElementById("step-db");
                if (stepDb) {
                    stepDb.className = "step-item completed d-flex align-items-center gap-2 mb-2 text-success small";
                    const icon = stepDb.querySelector("i, .step-icon");
                    if (icon) icon.className = "bi bi-check-circle-fill text-success";
                }
                const stepDbText = document.getElementById("step-db-text");
                if (stepDbText && data.request_id) {
                    stepDbText.textContent = `SQLite Request #${data.request_id} Logged`;
                }

                // Actively poll notification status for this request to ensure popup receives final delivery status
                this.startNotificationStatusPolling(data.request_id);
            }
        } catch (err) {
            console.error("Network error triggering block", err);
            this.releaseLockUI();
            this.showToast("Network dispatch error. Local alert active.", "danger");
            this.resetAllBlockDwells();
        }
    }

    /**
     * Visually engage the Lockout State on the interface
     */
    engageLockUI(blockId, title, priority = "MEDIUM", message = "", triggerSource = "gaze") {
        this.isLocked = true;
        this.lockedBlockId = blockId;
        this.lastDeliveryStatus = null;

        // Reset dwell counters and clear all card hover highlights
        this.resetAllBlockDwells();
        document.body.classList.add("lockout-active");
        if (this.cursorEl) {
            this.cursorEl.classList.add("lockout-hidden");
            this.cursorEl.style.display = "none";
            this.cursorEl.style.opacity = "0";
        }

        // Lock the blocks grid and highlight active locked card
        this.blocksGrid.classList.add("locked-mode");
        this.cards.forEach(c => {
            const cId = c.getAttribute("data-block-id");
            if (cId === blockId) {
                c.classList.add("active-locked-card", "locked");
            } else {
                c.classList.remove("active-locked-card", "locked");
            }
        });

        // Populate Lockout Overlay Modal
        const titleEl = document.getElementById("lockout-block-title");
        if (titleEl) titleEl.textContent = title || "REQUEST LOCKED";
        const badge = document.getElementById("lockout-priority-badge");
        if (badge) {
            badge.textContent = `${priority} PRIORITY`;
            const badgeClass = priority === "CRITICAL" ? "text-bg-danger" : (priority === "HIGH" ? "text-bg-warning" : "text-bg-primary");
            badge.className = `lockout-badge badge ${badgeClass} mb-3`;
        }

        const shieldIcon = document.getElementById("lock-shield-icon");
        if (shieldIcon) {
            shieldIcon.className = "bi bi-shield-lock-fill fs-1 text-warning";
        }

        const descEl = document.getElementById("lockout-desc");
        if (descEl) {
            descEl.textContent = `Communication request locked for "${title}". Dispatching alert to caregiver. All inputs are frozen.`;
        }

        // Reset step indicators
        const stepDwell = document.getElementById("step-dwell");
        const stepEmail = document.getElementById("step-email");
        const stepDb = document.getElementById("step-db");
        const stepDwellText = document.getElementById("step-dwell-text");
        const stepDbText = document.getElementById("step-db-text");

        if (stepDwell) {
            stepDwell.className = "step-item completed d-flex align-items-center gap-2 mb-2 text-success small";
            const icon = stepDwell.querySelector("i, .step-icon");
            if (icon) {
                icon.className = triggerSource === "keyboard" ? "bi bi-keyboard-fill" : "bi bi-check-circle-fill";
            }
        }
        if (stepDwellText) {
            stepDwellText.textContent = triggerSource === "keyboard" ? "Keyboard Shortcut Activated" : "Sustained Gaze Verified";
        }

        if (stepEmail) {
            stepEmail.className = "step-item active d-flex align-items-center gap-2 mb-2 text-warning small";
            const icon = stepEmail.querySelector("i, .step-icon");
            if (icon) icon.className = "bi bi-hourglass-split";
            const spinner = stepEmail.querySelector(".spinner-border");
            if (spinner) spinner.style.display = "inline-block";
        }

        if (stepDb) stepDb.className = "step-item d-flex align-items-center gap-2 text-secondary small";
        if (stepDbText) stepDbText.textContent = "SQLite Request Logged";

        const stepEmailText = document.getElementById("step-email-text");
        if (stepEmailText) stepEmailText.textContent = "Caregiver Email Dispatching...";

        // Show overlay modal
        this.lockoutModal.classList.add("active");

        // Cooldown bar animation (GA-5)
        if (this.cooldownFill) {
            this.cooldownFill.style.width = "0%";
            if (this.cooldownInterval) clearInterval(this.cooldownInterval);
            let progress = 0;
            this.cooldownInterval = setInterval(() => {
                progress += 2;
                if (progress <= 100 && this.cooldownFill) {
                    this.cooldownFill.style.width = `${progress}%`;
                }
            }, 60);
        }
    }

    /**
     * Release the Lockout State after delivery confirmation
     */
    releaseLockUI() {
        if (this._notifPollInterval) {
            clearInterval(this._notifPollInterval);
            this._notifPollInterval = null;
        }
        this._pollingRequestId = null;
        if (this._releaseTimeout) {
            clearTimeout(this._releaseTimeout);
            this._releaseTimeout = null;
        }

        this.isLocked = false;
        this.lockedBlockId = null;

        // Hide overlay modal
        this.lockoutModal.classList.remove("active");
        document.body.classList.remove("lockout-active");
        if (this.cursorEl) {
            this.cursorEl.classList.remove("lockout-hidden");
        }
        this.resetAllBlockDwells();

        // Complete and reset cooldown bar (GA-5)
        if (this.cooldownInterval) {
            clearInterval(this.cooldownInterval);
            this.cooldownInterval = null;
        }
        if (this.cooldownFill) {
            this.cooldownFill.style.width = "100%";
            setTimeout(() => {
                if (!this.isLocked && this.cooldownFill) {
                    this.cooldownFill.style.width = "0%";
                }
            }, 300);
        }

        // Unlock blocks grid
        this.blocksGrid.classList.remove("locked-mode");
        this.cards.forEach(c => {
            c.classList.remove("active-locked-card", "locked");
            const bId = c.getAttribute("data-block-id");
            this.resetBlockDwell(bId);
        });

        // Set state to WAITING_FOR_RELEASE:
        // Gaze must leave the selected card before any new card can be selected
        this.selectionState = 'WAITING_FOR_RELEASE';
        this.activeHoverBlock = null;
        this.dwellStartTime = null;
        this.lastDwellTickWindow = -1;

        if (this.lastDeliveryStatus === "DELIVERED") {
            this.playChimeDelivered();
            this.showToast("System Unlocked: Alert delivered to caregiver.");
        } else if (this.lastDeliveryStatus === "SIMULATED") {
            this.showToast("System Unlocked: Alert recorded (Simulation Mode active).");
        } else if (this.lastDeliveryStatus === "FAILED") {
            this.showToast("System Unlocked: Email delivery failed. Please check caregiver settings.");
        } else {
            this.showToast("System Unlocked: Look away from card to prepare next selection.");
        }
        this.lastDeliveryStatus = null;
    }

    /**
     * Synchronize server-side lock state with client UI and update popup status.
     */
    syncLockState(lock) {
        if (!lock) return;

        if (lock.is_locked) {
            if (!this.isLocked) {
                this.engageLockUI(
                    lock.locked_block_id || "request",
                    lock.locked_block_title || "Assistance Request",
                    lock.priority || "MEDIUM",
                    lock.message || "",
                    lock.trigger_source || "gaze"
                );
            }
            this.updateLockoutNotificationStatus(lock);
        } else if (this.isLocked && !lock.is_locked) {
            if (lock.email_status || lock.notification_status) {
                this.updateLockoutNotificationStatus(lock);
            }
            if (this._releaseTimeout) clearTimeout(this._releaseTimeout);
            this._releaseTimeout = setTimeout(() => {
                if (this.isLocked) {
                    this.releaseLockUI();
                }
            }, 600);
        }
    }

    /**
     * Update lockout overlay modal with real-time delivery status
     */
    updateLockoutNotificationStatus(lock) {
        if (!lock) return;

        const emailStatus = lock.email_status || lock.notification_status;
        const requestId = lock.last_request_id || lock.request_id;
        const message = lock.message || "";

        // Update Step 3 (Database Logged) if request ID is known
        const stepDb = document.getElementById("step-db");
        const stepDbText = document.getElementById("step-db-text");
        if (stepDb && requestId) {
            stepDb.className = "step-item completed d-flex align-items-center gap-2 mb-2 text-success small";
            const icon = stepDb.querySelector("i, .step-icon");
            if (icon) icon.className = "bi bi-check-circle-fill text-success";
            if (stepDbText) stepDbText.textContent = `SQLite Request #${requestId} Logged`;
        }

        const stepEmail = document.getElementById("step-email");
        const stepEmailText = document.getElementById("step-email-text");
        const lockoutDesc = document.getElementById("lockout-desc");
        const shieldIcon = document.getElementById("lock-shield-icon");
        const badge = document.getElementById("lockout-priority-badge");

        if (!stepEmail || !emailStatus) return;

        const spinner = stepEmail.querySelector(".spinner-border");
        let icon = stepEmail.querySelector("i, .step-icon");
        if (!icon) {
            icon = document.createElement("i");
            icon.className = "step-icon";
            stepEmail.prepend(icon);
        }

        if (emailStatus === "DELIVERED") {
            this.lastDeliveryStatus = "DELIVERED";
            stepEmail.className = "step-item completed d-flex align-items-center gap-2 mb-2 text-success small";
            if (spinner) spinner.style.display = "none";
            icon.style.display = "inline-block";
            icon.className = "bi bi-check-circle-fill text-success";
            if (stepEmailText) {
                const durStr = lock.duration_ms ? ` (${lock.duration_ms}ms)` : "";
                stepEmailText.textContent = `Alert Delivered to Caregiver (via Brevo API${durStr})`;
            }
            if (lockoutDesc) lockoutDesc.textContent = "Alert successfully dispatched to caregiver. Safety lock will release momentarily.";
            if (shieldIcon) shieldIcon.className = "bi bi-shield-check fs-1 text-success";
            if (badge) {
                badge.className = "lockout-badge badge text-bg-success mb-3";
                badge.textContent = "ALERT DELIVERED";
            }
        } else if (emailStatus === "SIMULATED") {
            this.lastDeliveryStatus = "SIMULATED";
            stepEmail.className = "step-item completed d-flex align-items-center gap-2 mb-2 text-info small";
            if (spinner) spinner.style.display = "none";
            icon.style.display = "inline-block";
            icon.className = "bi bi-info-circle-fill text-info";
            if (stepEmailText) stepEmailText.textContent = "Simulation Mode (No Real Email Sent)";
            if (lockoutDesc) lockoutDesc.textContent = "Alert recorded in simulation mode. Safety lock will release momentarily.";
            if (shieldIcon) shieldIcon.className = "bi bi-shield-check fs-1 text-info";
            if (badge) {
                badge.className = "lockout-badge badge text-bg-info mb-3";
                badge.textContent = "SIMULATED ALERT";
            }
        } else if (emailStatus === "FAILED") {
            this.lastDeliveryStatus = "FAILED";
            stepEmail.className = "step-item error d-flex align-items-center gap-2 mb-2 text-danger small";
            if (spinner) spinner.style.display = "none";
            icon.style.display = "inline-block";
            icon.className = "bi bi-exclamation-triangle-fill text-danger";
            if (stepEmailText) {
                stepEmailText.textContent = message ? `Delivery Failed: ${message}` : "Caregiver Email Delivery Failed";
            }
            if (lockoutDesc) lockoutDesc.textContent = message || "Email dispatch failed. Please check caregiver settings. Releasing...";
            if (shieldIcon) shieldIcon.className = "bi bi-shield-slash-fill fs-1 text-danger";
            if (badge) {
                badge.className = "lockout-badge badge text-bg-danger mb-3";
                badge.textContent = "DELIVERY FAILED";
            }
        } else if (emailStatus === "SENDING") {
            stepEmail.className = "step-item active d-flex align-items-center gap-2 mb-2 text-warning small";
            if (spinner) spinner.style.display = "inline-block";
            icon.style.display = "none";
            if (stepEmailText) stepEmailText.textContent = "Caregiver Email Dispatching...";
        }
    }

    /**
     * Start fallback polling for notification status while locked.
     */
    startNotificationStatusPolling(requestId) {
        if (this._notifPollInterval) {
            clearInterval(this._notifPollInterval);
            this._notifPollInterval = null;
        }
        this._pollingRequestId = requestId;
        let pollCount = 0;
        this._notifPollInterval = setInterval(async () => {
            pollCount++;
            if (!this.isLocked || pollCount > 60) {
                clearInterval(this._notifPollInterval);
                this._notifPollInterval = null;
                return;
            }
            try {
                const res = await fetch(`/api/notification_status?request_id=${encodeURIComponent(requestId || "")}`);
                if (!res.ok) return;
                const statusData = await res.json();
                if (statusData && statusData.email_status) {
                    this.updateLockoutNotificationStatus(statusData);
                    if (["DELIVERED", "SIMULATED", "FAILED"].includes(statusData.email_status)) {
                        if (!statusData.is_locked) {
                            clearInterval(this._notifPollInterval);
                            this._notifPollInterval = null;
                            setTimeout(() => {
                                if (this.isLocked) {
                                    this.releaseLockUI();
                                }
                            }, 600);
                        }
                    }
                }
            } catch (e) { }
        }, 350);
    }

    /**
     * Force unlock override
     */
    async forceUnlock() {
        try {
            await fetch("/api/unlock", { method: "POST", headers: this.getHeaders() });
            this.releaseLockUI();
        } catch (e) {
            this.releaseLockUI();
        }
    }

    /**
     * Trigger Re-center baseline gaze
     */
    async triggerRecenter() {
        this.playTone(600, "sine", 0.15, 0.1);
        try {
            await fetch("/api/recenter", { method: "POST", headers: this.getHeaders() });
            this.showToast("Baseline Re-Centered to Current Resting Pose");
        } catch (e) {
            this.showToast("Re-center triggered.");
        }
    }

    /**
     * Load settings configuration from server
     */
    async loadConfig() {
        try {
            const res = await fetch("/api/config");
            const cfg = await res.json();

            this.dwellDuration = (cfg.dwell_time_seconds || 3.0) * 1000;
            this.soundEnabled = Boolean(cfg.sound_enabled);
            this.voiceEnabled = false; // Voice assistance disabled per user request

            // Update header labels
            this.currentDwellLabel.textContent = `${cfg.dwell_time_seconds || 3.0}s`;
            document.getElementById("patient-name-display").textContent = cfg.patient_name || "Patient";
            document.getElementById("caregiver-email-display").textContent = cfg.caregiver_email || "caregiver@example.com";

            // Update form tracking mode
            const trackModeSelect = document.getElementById("cfg-tracking-mode");
            if (trackModeSelect) {
                trackModeSelect.value = cfg.tracking_mode || "both_eyes";
            }

            if (cfg.calibration_dwell_seconds !== undefined) {
                this.calibrationDwellDuration = parseFloat(cfg.calibration_dwell_seconds) * 1000;
                const calibSlider = document.getElementById("cfg-calib-dwell-time");
                if (calibSlider) calibSlider.value = cfg.calibration_dwell_seconds;
                const calibLabel = document.getElementById("val-calib-dwell-time");
                if (calibLabel) calibLabel.textContent = `${cfg.calibration_dwell_seconds}s`;
            }

            // Update simulation mode toggle, badge and text
            const simCheckbox = document.getElementById("cfg-simulation-mode");
            if (simCheckbox) {
                simCheckbox.checked = Boolean(cfg.simulation_mode);
            }
            this.updateSimulationModeUI(Boolean(cfg.simulation_mode));

            // Populate Brevo API credentials
            const apiKeyInput = document.getElementById("cfg-brevo-api-key");
            const modalStatus = document.getElementById("modal-brevo-key-status");
            if (apiKeyInput) {
                apiKeyInput.value = "";
                if (cfg.has_brevo_api_key) {
                    apiKeyInput.placeholder = "Leave blank to keep saved key";
                    if (modalStatus) {
                        modalStatus.textContent = "Brevo API key configured";
                        modalStatus.className = "badge text-bg-success";
                    }
                } else {
                    apiKeyInput.placeholder = "Enter Brevo API key (xkeysib-...)";
                    if (modalStatus) {
                        modalStatus.textContent = "No API key saved";
                        modalStatus.className = "badge text-bg-secondary";
                    }
                }
            }
            const senderNameInput = document.getElementById("cfg-brevo-sender-name");
            if (senderNameInput && cfg.brevo_sender_name) {
                senderNameInput.value = cfg.brevo_sender_name;
            }
            const senderEmailInput = document.getElementById("cfg-brevo-sender-email");
            if (senderEmailInput && cfg.brevo_sender_email) {
                senderEmailInput.value = cfg.brevo_sender_email;
            }

            // Sync camera index & mirror view
            if (cfg.camera_index !== undefined) {
                this.cameraIndex = parseInt(cfg.camera_index, 10);
            }
            if (cfg.mirror_view !== undefined) {
                this.mirrorView = Boolean(cfg.mirror_view);
                this.updateMirrorUI();
            }

            const useTlsSelect = document.getElementById("cfg-use-tls");
            if (useTlsSelect && cfg.use_tls !== undefined) {
                useTlsSelect.value = String(cfg.use_tls);
            }

            this.populateCameraOptions();
        } catch (e) {
            console.error("Config fetch error", e);
        }
    }

    /**
     * Save settings configuration
     */
    async saveSettings() {
        const form = document.getElementById("settings-form");
        const formData = new FormData(form);

        const payload = {
            patient_name: formData.get("patient_name"),
            caregiver_email: formData.get("caregiver_email"),
            caregiver_phone: formData.get("caregiver_phone"),
            dwell_time_seconds: parseFloat(formData.get("dwell_time_seconds")),
            calibration_dwell_seconds: parseFloat(formData.get("calibration_dwell_seconds") || 1.5),
            head_sensitivity: parseFloat(formData.get("head_sensitivity")),
            eye_sensitivity: parseFloat(formData.get("eye_sensitivity")),
            tracking_mode: formData.get("tracking_mode") || "both_eyes",
            camera_index: parseInt(formData.get("camera_index") !== null ? formData.get("camera_index") : this.cameraIndex, 10),
            mirror_view: parseInt(formData.get("mirror_view") !== null ? formData.get("mirror_view") : (this.mirrorView ? 1 : 0), 10),
            sound_enabled: document.getElementById("cfg-sound-enabled").checked ? 1 : 0,
            voice_enabled: 0,
            simulation_mode: document.getElementById("cfg-simulation-mode").checked ? 1 : 0,
            email_provider: "brevo",
            brevo_sender_name: formData.get("brevo_sender_name") || "Eye Gaze AI",
            brevo_sender_email: formData.get("brevo_sender_email") || "ogjoyboy21@gmail.com"
        };

        const apiKeyVal = formData.get("brevo_api_key");
        if (apiKeyVal && !apiKeyVal.includes("•")) {
            payload.brevo_api_key = apiKeyVal.trim();
        }

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (data.success) {
                this.showToast("Configuration Saved Successfully!");
                this.dwellDuration = payload.dwell_time_seconds * 1000;
                this.soundEnabled = Boolean(payload.sound_enabled);
                this.voiceEnabled = Boolean(payload.voice_enabled);
                this.currentDwellLabel.textContent = `${payload.dwell_time_seconds}s`;
                this.calibrationDwellDuration = payload.calibration_dwell_seconds * 1000;
                document.getElementById("patient-name-display").textContent = payload.patient_name;
                document.getElementById("caregiver-email-display").textContent = payload.caregiver_email;
                this.updateSimulationModeUI(Boolean(payload.simulation_mode));

                // Update Brevo API key input & status badge
                const apiKeyInput = document.getElementById("cfg-brevo-api-key");
                const modalStatus = document.getElementById("modal-brevo-key-status");
                const hasKey = (data.config && data.config.has_brevo_api_key) || Boolean(payload.brevo_api_key);
                if (apiKeyInput) {
                    apiKeyInput.value = "";
                    if (hasKey) {
                        apiKeyInput.placeholder = "Leave blank to keep saved key";
                        if (modalStatus) {
                            modalStatus.textContent = "Brevo API key configured";
                            modalStatus.className = "badge text-bg-success";
                        }
                    }
                }

                this.settingsModal.classList.add("hidden");
            }
        } catch (err) {
            this.showToast("Error saving configuration.");
        }
    }

    /**
     * Send test email
     */
    async sendTestEmail() {
        const feedback = document.getElementById("test-email-feedback");
        feedback.className = "test-feedback";
        feedback.textContent = "Connecting to Brevo API...";

        const email = document.getElementById("cfg-caregiver-email").value.trim();
        const apiKey = document.getElementById("cfg-brevo-api-key").value;
        const senderName = document.getElementById("cfg-brevo-sender-name").value.trim();
        const senderEmail = document.getElementById("cfg-brevo-sender-email").value.trim();
        const simMode = document.getElementById("cfg-simulation-mode").checked ? 1 : 0;

        const payload = {
            caregiver_email: email,
            email_provider: "brevo",
            brevo_sender_name: senderName,
            brevo_sender_email: senderEmail,
            simulation_mode: simMode
        };
        if (apiKey && !apiKey.includes("•")) {
            payload.brevo_api_key = apiKey.trim();
        }

        try {
            const res = await fetch("/api/test_email", {
                method: "POST",
                headers: this.getHeaders(),
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.status === "DELIVERED") {
                feedback.className = "small text-success fw-bold";
                feedback.textContent = `✅ Live email accepted by Brevo for ${email} (${data.duration_ms}ms)! Check your inbox.`;
            } else if (data.status === "SIMULATED") {
                feedback.className = "small text-warning fw-bold";
                feedback.textContent = `ℹ️ Simulation mode — no real email is sent (${data.message || ""})`;
            } else {
                feedback.className = "small text-danger fw-bold";
                feedback.textContent = `❌ Delivery rejected: ${data.message}`;
            }
        } catch (e) {
            feedback.className = "test-feedback error";
            feedback.textContent = "Network error testing email.";
        }
    }

    async openLogs() {
        this.logsModal.classList.remove("hidden");
        const tbody = document.getElementById("logs-table-body");
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Loading history logs...</td></tr>';

        try {
            const res = await fetch("/api/logs?limit=50");

            const data = await res.json();
            if (data.success && data.logs && data.logs.length > 0) {
                tbody.innerHTML = "";
                data.logs.forEach(log => {
                    const tr = document.createElement("tr");
                    const dateStr = log.timestamp || "Recently";
                    const priorityClass = log.priority === "CRITICAL" ? "text-bg-danger" : (log.priority === "HIGH" ? "text-bg-warning" : "text-bg-primary");
                    const statusClass = log.status === "DELIVERED" ? "text-bg-success" : (log.status === "SIMULATED" ? "text-bg-info" : (log.status === "PENDING" || log.status === "SENDING" ? "text-bg-warning" : "text-bg-danger"));
                    tr.innerHTML = `
                        <td class="font-monospace small text-secondary">${dateStr}</td>
                        <td><strong>${log.block_title}</strong> <span class="text-secondary small">(${log.category})</span></td>
                        <td><span class="badge ${priorityClass}">${log.priority}</span></td>
                        <td class="small font-monospace text-secondary">${log.caregiver_email}</td>
                        <td><span class="badge ${statusClass}">${log.status}</span></td>
                        <td class="small font-monospace text-secondary">${log.delivery_duration_ms} ms</td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No requests logged yet. Use eye-gaze or hotkeys 1-4 to trigger requests.</td></tr>';
            }
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Failed to fetch logs.</td></tr>';
        }
    }

    /**
     * Clear all logs
     */
    async clearLogs() {
        const confirmed = await new Promise((resolve) => {
            const modal = document.createElement("div");
            modal.className = "legacy-modal-overlay";
            modal.style.zIndex = "10001";
            modal.innerHTML = `
                <div class="modal-window" style="max-width: 440px;">
                    <div class="modal-header">
                        <div class="modal-title-group">
                            <i class="bi bi-trash3-fill text-danger fs-5"></i>
                            <h2>Clear History Logs</h2>
                        </div>
                        <button class="modal-close-btn" id="confirm-modal-close" aria-label="Close">
                            <i class="bi bi-x-lg"></i>
                        </button>
                    </div>
                    <div class="modal-body" style="padding: 20px 22px;">
                        <p style="margin: 0 0 20px 0; color: var(--text-secondary); font-size: 14px; line-height: 1.5;">
                            Are you sure you want to clear all patient request history? This action cannot be undone.
                        </p>
                        <div class="modal-footer" style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 0;">
                            <button type="button" class="action-btn secondary" id="confirm-modal-cancel">Cancel</button>
                            <button type="button" class="action-btn danger" id="confirm-modal-ok">
                                <i class="bi bi-trash-fill me-1"></i>
                                <span>Clear Logs</span>
                            </button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);

            const cleanup = (result) => {
                document.removeEventListener("keydown", handleKeydown, true);
                modal.remove();
                resolve(result);
            };

            const handleKeydown = (e) => {
                if (e.key === "Escape") {
                    e.stopPropagation();
                    e.preventDefault();
                    cleanup(false);
                }
            };
            document.addEventListener("keydown", handleKeydown, true);

            modal.querySelector("#confirm-modal-ok").addEventListener("click", () => cleanup(true));
            modal.querySelector("#confirm-modal-cancel").addEventListener("click", () => cleanup(false));
            modal.querySelector("#confirm-modal-close").addEventListener("click", () => cleanup(false));
            modal.addEventListener("click", (e) => {
                if (e.target === modal) cleanup(false);
            });

            modal.querySelector("#confirm-modal-cancel").focus();
        });

        if (!confirmed) return;

        try {
            const res = await fetch("/api/logs/clear", { method: "POST", headers: this.getHeaders() });

            if (this.logsModal && !this.logsModal.classList.contains("hidden")) {
                this.openLogs();
            }
            this.showToast("History logs cleared.", "success");
        } catch (e) { }
    }

    /**
     * Open settings modal
     */
    async openSettings() {
        try {
            const res = await fetch("/api/config");
            const cfg = await res.json();
            if (cfg) {
                const elPatient = document.getElementById("cfg-patient-name");
                if (elPatient && cfg.patient_name) elPatient.value = cfg.patient_name;
                const elEmail = document.getElementById("cfg-caregiver-email");
                if (elEmail && cfg.caregiver_email) elEmail.value = cfg.caregiver_email;
                const elPhone = document.getElementById("cfg-caregiver-phone");
                if (elPhone && cfg.caregiver_phone) elPhone.value = cfg.caregiver_phone;

                // Brevo API credentials in settings modal
                const apiKeyInput = document.getElementById("cfg-brevo-api-key");
                const modalStatus = document.getElementById("modal-brevo-key-status");
                if (apiKeyInput) {
                    apiKeyInput.value = "";
                    if (cfg.has_brevo_api_key) {
                        apiKeyInput.placeholder = "Leave blank to keep saved key";
                        if (modalStatus) {
                            modalStatus.textContent = "Brevo API key configured";
                            modalStatus.className = "badge text-bg-success";
                        }
                    } else {
                        apiKeyInput.placeholder = "Enter Brevo API key (xkeysib-...)";
                        if (modalStatus) {
                            modalStatus.textContent = "No API key saved";
                            modalStatus.className = "badge text-bg-secondary";
                        }
                    }
                }
                const senderNameInput = document.getElementById("cfg-brevo-sender-name");
                if (senderNameInput && cfg.brevo_sender_name) {
                    senderNameInput.value = cfg.brevo_sender_name;
                }
                const senderEmailInput = document.getElementById("cfg-brevo-sender-email");
                if (senderEmailInput && cfg.brevo_sender_email) {
                    senderEmailInput.value = cfg.brevo_sender_email;
                }

                if (this.cfgCamSelect && cfg.camera_index !== undefined) {
                    this.cfgCamSelect.value = String(cfg.camera_index);
                }
                const cfgMirror = document.getElementById("cfg-mirror-view");
                if (cfgMirror && cfg.mirror_view !== undefined) {
                    cfgMirror.value = cfg.mirror_view ? "1" : "0";
                }
                const trackMode = document.getElementById("cfg-tracking-mode");
                if (trackMode && cfg.tracking_mode) {
                    trackMode.value = cfg.tracking_mode;
                }
            }
        } catch (e) { }
        this.settingsModal.classList.remove("hidden");
    }

    /**
     * Close all open modals
     */
    closeAllModals() {
        this.logsModal.classList.add("hidden");
        this.settingsModal.classList.add("hidden");
    }

    /**
     * Display temporary visual toast message
     * Supports ARIA accessibility, typed styles (warning, danger, success, info), and safe durations
     */
    showToast(msg, typeOrDuration = null, optionalDuration = null) {
        if (!this.toastContainer) return;
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = msg;

        let toastType = "info";
        let duration = null;

        if (typeof typeOrDuration === "number") {
            duration = typeOrDuration;
        } else if (typeof typeOrDuration === "string") {
            toastType = typeOrDuration.toLowerCase();
            duration = (typeof optionalDuration === "number") ? optionalDuration : null;
        }

        const isCritical = /⚠️|error|failed|expired|denied|locked/i.test(msg) || ["error", "danger", "warning"].includes(toastType);
        if (toastType === "error" || toastType === "danger") {
            toast.classList.add("toast-danger");
        } else if (toastType === "warning") {
            toast.classList.add("toast-warning");
        } else if (toastType === "success") {
            toast.classList.add("toast-success");
        } else if (toastType === "info") {
            toast.classList.add("toast-info");
        }

        // Accessibility attributes (AM-2)
        toast.setAttribute("role", isCritical ? "alert" : "status");
        toast.setAttribute("aria-live", isCritical ? "assertive" : "polite");

        // Allow click to dismiss immediately (AM-3)
        toast.style.cursor = "pointer";
        toast.title = "Click to dismiss";

        const displayTime = duration || (isCritical ? 5000 : 3500);

        const removeToast = () => {
            if (toast._dismissed) return;
            toast._dismissed = true;
            toast.style.opacity = "0";
            toast.style.transform = "translateX(20px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        };

        toast.addEventListener("click", removeToast);
        this.toastContainer.appendChild(toast);
        setTimeout(removeToast, displayTime);
    }
}

// Instantiate application upon DOM load
document.addEventListener("DOMContentLoaded", () => {
    window.app = new GazeAssistant();
});
