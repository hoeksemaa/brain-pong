// BrainPong — clientside renderer.
//
// Architecture: a self-bootstrapping requestAnimationFrame loop owns all
// drawing. Dash clientside callbacks push the latest game/app state into a
// closure-scoped object; the rAF loop reads it on every frame. Flicker is
// frame-counted (not time-computed), so phase is deterministic relative to
// the display refresh, with no setInterval / callback-fire jitter.
//
// Targets a 120 Hz display (Chrome + macOS ProMotion). Measures the actual
// rAF rate over the first ~120 frames, picks the nearest clean refresh from
// {60, 90, 120}, and computes integer frame periods for each stimulus
// frequency. Falls loud (on-canvas warning + console.warn) if measurement
// is outside tolerance.

if (!window.dash_clientside) { window.dash_clientside = {}; }

(function () {
    "use strict";

    // ---------------- Layout (mirrors Python GAME_* constants) ----------------
    const PADDLE_HEIGHT = 20;
    const BALL_RADIUS = 10;
    const BAND_FRACTION = 0.25;

    // ---------------- Colors ----------------
    const COL_FLICKER_ON  = '#FFFFFF';
    const COL_FLICKER_OFF = '#000000';
    const COL_BG_CENTER   = '#1a1a1a';
    const COL_PADDLE_PLAYER = '#33ff66';
    const COL_PADDLE_AI     = '#ff5252';
    const COL_BALL          = '#ffe600';
    const COL_SCORE         = '#FFFFFF';
    const COL_WARN          = '#ff7777';

    // ---------------- Refresh-rate detection ----------------
    const TARGET_REFRESH_HZ = 120;
    const REFRESH_CANDIDATES = [60, 90, 120];
    const MEASUREMENT_FRAMES = 120;
    const REFRESH_TOLERANCE_HZ = 3;

    // ---------------- Mutable state pushed by Dash ----------------
    const dashState = {
        gameState: null,
        appStatus: null,
        settings: null,
        noBoard: false,
        freqLeft: 10,
        freqRight: 15,
        canvasId: null,
    };

    // ---------------- Loop bookkeeping ----------------
    let started = false;
    let frameIdx = 0;
    let lastEdgeStateLeft = null;
    let lastEdgeStateRight = null;

    // Refresh measurement
    const measureDeltas = [];
    let lastFrameMs = null;
    let measuredHz = null;
    let chosenRefreshHz = TARGET_REFRESH_HZ;
    let measuredOk = false;
    let measurementWarning = null;
    let measurementDone = false;

    // Frame periods (set after measurement; safe pre-measurement defaults)
    let leftPeriodFrames = 12;
    let rightPeriodFrames = 8;

    // Edge log (ring buffer; for future recording mode to consume)
    const EDGE_LOG_MAX = 4096;
    const edgeLog = [];

    // Visibility tracking
    let tabVisible = true;
    document.addEventListener('visibilitychange', function () {
        tabVisible = (document.visibilityState === 'visible');
        if (!tabVisible) {
            console.warn('[brainpong] tab hidden — rAF will throttle. Recording integrity at risk.');
        }
        window.dash_clientside.brainpong_tab_visible = tabVisible;
    });

    // ---------------- Period math ----------------
    function recomputePeriods(refreshHz, freqL, freqR) {
        function evenPeriod(refresh, freq) {
            let p = Math.round(refresh / freq);
            if (p < 2) p = 2;
            if (p % 2 !== 0) p += 1; // force even so on/off halves are equal
            return p;
        }
        leftPeriodFrames  = evenPeriod(refreshHz, freqL);
        rightPeriodFrames = evenPeriod(refreshHz, freqR);
    }

    function actualHz(refreshHz, periodFrames) {
        return refreshHz / periodFrames;
    }

    // ---------------- Refresh measurement ----------------
    function finalizeMeasurement() {
        if (measurementDone) return;
        measurementDone = true;

        const sorted = measureDeltas.slice().sort(function (a, b) { return a - b; });
        const median = sorted[Math.floor(sorted.length / 2)];
        measuredHz = 1000 / median;

        let chosen = REFRESH_CANDIDATES[0];
        let bestDiff = Infinity;
        for (let i = 0; i < REFRESH_CANDIDATES.length; i++) {
            const diff = Math.abs(REFRESH_CANDIDATES[i] - measuredHz);
            if (diff < bestDiff) { bestDiff = diff; chosen = REFRESH_CANDIDATES[i]; }
        }

        measuredOk = bestDiff < REFRESH_TOLERANCE_HZ;
        chosenRefreshHz = chosen;

        if (!measuredOk) {
            measurementWarning = 'unstable refresh: measured ' + measuredHz.toFixed(1) +
                ' Hz, no clean match in {60, 90, 120}';
            console.warn('[brainpong] ' + measurementWarning);
        } else {
            console.log('[brainpong] refresh detected ~' + measuredHz.toFixed(2) +
                        ' Hz — using ' + chosen + ' Hz frame counts');
            recomputePeriods(chosen, dashState.freqLeft, dashState.freqRight);
        }

        window.dash_clientside.brainpong_measurement = {
            measuredHz: measuredHz,
            chosenRefreshHz: chosenRefreshHz,
            ok: measuredOk,
            warning: measurementWarning,
            leftPeriodFrames: leftPeriodFrames,
            rightPeriodFrames: rightPeriodFrames,
            actualLeftHz: actualHz(chosenRefreshHz, leftPeriodFrames),
            actualRightHz: actualHz(chosenRefreshHz, rightPeriodFrames),
        };
    }

    // ---------------- Edge log ----------------
    function logEdge(side, isOn, frame, ms) {
        if (edgeLog.length >= EDGE_LOG_MAX) edgeLog.shift();
        edgeLog.push({ side: side, isOn: isOn, frame: frame, ms: ms });
    }

    // ---------------- Drawing ----------------
    function flickerAllowed(status, side) {
        // Flicker runs in both --no-board and real-hardware modes; the SSVEP
        // stimulus is the experience, and dev/test needs visual verification.
        if (!status) return false;
        if (status === 'PLAYING' || status === 'READY') return true;
        if (status.indexOf('CALIBRATING_REST')  !== -1) return true;
        if (side === 'L' && status.indexOf('CALIBRATING_LEFT')  !== -1) return true;
        if (side === 'R' && status.indexOf('CALIBRATING_RIGHT') !== -1) return true;
        return false;
    }

    function draw(ctx, W, H, isLeftOn, isRightOn) {
        const bandW = W * BAND_FRACTION;
        const rightBandX = W * (1 - BAND_FRACTION);

        ctx.fillStyle = COL_BG_CENTER;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = isLeftOn ? COL_FLICKER_ON : COL_FLICKER_OFF;
        ctx.fillRect(0, 0, bandW, H);
        ctx.fillStyle = isRightOn ? COL_FLICKER_ON : COL_FLICKER_OFF;
        ctx.fillRect(rightBandX, 0, bandW, H);

        const gs = dashState.gameState;
        if (gs) {
            const paddleW = (dashState.settings && dashState.settings.paddle_width) || 150;
            ctx.fillStyle = COL_PADDLE_AI;
            ctx.fillRect(gs.ai_x - paddleW / 2, 0, paddleW, PADDLE_HEIGHT);
            ctx.fillStyle = COL_PADDLE_PLAYER;
            ctx.fillRect(gs.player_x - paddleW / 2, H - PADDLE_HEIGHT, paddleW, PADDLE_HEIGHT);

            ctx.fillStyle = COL_BALL;
            ctx.beginPath();
            ctx.arc(gs.ball_x, gs.ball_y, BALL_RADIUS, 0, 2 * Math.PI);
            ctx.fill();

            ctx.font = 'bold 40px ui-monospace, Menlo, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = COL_SCORE;
            ctx.fillText(String(gs.ai_score), W / 2, 12);
            ctx.textBaseline = 'bottom';
            ctx.fillText(String(gs.player_score), W / 2, H - PADDLE_HEIGHT - 8);
        }

        if (measurementWarning) {
            ctx.font = '13px ui-monospace, Menlo, monospace';
            ctx.fillStyle = COL_WARN;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'bottom';
            ctx.fillText('⚠ ' + measurementWarning, bandW + 10, H - 6);
        }
        if (!tabVisible) {
            ctx.font = '13px ui-monospace, Menlo, monospace';
            ctx.fillStyle = COL_WARN;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('⚠ tab not visible', bandW + 10, 6);
        }
    }

    // ---------------- Main loop ----------------
    function loop(nowMs) {
        // Frame-delta sampling for refresh measurement
        if (lastFrameMs !== null) {
            const d = nowMs - lastFrameMs;
            if (measureDeltas.length < MEASUREMENT_FRAMES) {
                measureDeltas.push(d);
                if (measureDeltas.length === MEASUREMENT_FRAMES) finalizeMeasurement();
            }
        }
        lastFrameMs = nowMs;

        const canvas = dashState.canvasId ? document.getElementById(dashState.canvasId) : null;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            const W = canvas.width;
            const H = canvas.height;

            const status = (dashState.appStatus && dashState.appStatus.status) || 'STARTING';
            const allowL = flickerAllowed(status, 'L');
            const allowR = flickerAllowed(status, 'R');

            const halfL = leftPeriodFrames / 2;
            const halfR = rightPeriodFrames / 2;
            const phaseL = Math.floor(frameIdx / halfL) % 2 === 0;
            const phaseR = Math.floor(frameIdx / halfR) % 2 === 0;

            const isLeftOn  = allowL && phaseL;
            const isRightOn = allowR && phaseR;

            if (allowL) {
                if (lastEdgeStateLeft !== isLeftOn) {
                    logEdge('L', isLeftOn, frameIdx, nowMs);
                    lastEdgeStateLeft = isLeftOn;
                }
            } else {
                lastEdgeStateLeft = null;
            }
            if (allowR) {
                if (lastEdgeStateRight !== isRightOn) {
                    logEdge('R', isRightOn, frameIdx, nowMs);
                    lastEdgeStateRight = isRightOn;
                }
            } else {
                lastEdgeStateRight = null;
            }

            draw(ctx, W, H, isLeftOn, isRightOn);
        }

        frameIdx++;
        requestAnimationFrame(loop);
    }

    // ---------------- Public entry point ----------------
    // Dash clientside_callback calls this whenever any of the input stores
    // change. First call boots the rAF loop; subsequent calls just refresh
    // the state mirror.
    function renderPong(canvasId, gameState, appStatus, settings, noBoard, freqLeft, freqRight) {
        dashState.canvasId  = canvasId;
        dashState.gameState = gameState;
        dashState.appStatus = appStatus;
        dashState.settings  = settings;
        dashState.noBoard   = !!noBoard;
        dashState.freqLeft  = freqLeft;
        dashState.freqRight = freqRight;

        if (!started) {
            started = true;
            recomputePeriods(TARGET_REFRESH_HZ, freqLeft, freqRight);
            requestAnimationFrame(loop);
        } else if (measurementDone && measuredOk) {
            // Re-evaluate periods if Dash ever pushes new freqs at runtime.
            recomputePeriods(chosenRefreshHz, freqLeft, freqRight);
        }
    }

    window.dash_clientside.renderPong = renderPong;
    window.dash_clientside.brainpong_edge_log = edgeLog;
    window.dash_clientside.brainpong_tab_visible = true;
})();
