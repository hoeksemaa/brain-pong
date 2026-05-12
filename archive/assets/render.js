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
        record: null,   // {recordingMode, totalTrials, trialDurationS, restDurationS, recState, saveStatus}
    };

    // Track when status enters RECORD_TRIAL so we can render a countdown bar
    // anchored to that moment (independent of the server-side 500ms tick).
    let trialPhaseStartMs = null;
    let restPhaseStartMs  = null;
    let lastObservedStatus = null;

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

    // Edge log (ring buffer). Sized to cover ~22 min of two-band flicker
    // (~50 transitions/sec). Recording mode consumes the whole array on
    // RECORD_DONE; in play mode this is a debug aid only.
    const EDGE_LOG_MAX = 65536;
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
        // Recording mode: flicker bands run continuously through the whole
        // session (READY / TRIAL / REST) so the SSVEP stimulus is uninterrupted.
        if (status === 'RECORD_READY' || status === 'RECORD_TRIAL' ||
            status === 'RECORD_REST'  || status === 'RECORD_DONE'  ||
            status === 'RECORD_SAVED') return true;
        return false;
    }

    function isRecordingStatus(status) {
        return status && status.indexOf('RECORD_') === 0;
    }

    // Adaptive y-limits per render call. Smoothed across frames so the scale
    // doesn't twitch on every update — same pattern as filtered_plot.py.
    const eegYLimits = { min: -50, max: 50 };

    function drawEegPlot(ctx, x0, y0, w, h, live) {
        // Draw a small multi-channel oscilloscope. Filtered, μV-scale,
        // adaptive y-axis. X axis = time (s, with 0 = "now"). Y = μV.
        ctx.save();
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(x0, y0, w, h);
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);

        if (!live || !live.streaming || !live.traces || !live.traces.length) {
            ctx.fillStyle = '#888';
            ctx.font = '12px ui-monospace, Menlo, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(live && live.reason ? '✗ ' + live.reason : 'waiting for EEG…',
                         x0 + w / 2, y0 + h / 2);
            ctx.restore();
            return;
        }

        const traces = live.traces;
        const nCh = traces.length;
        const windowS = live.window_s || 1.5;

        // Per-channel auto-scale across all visible samples + 20% padding.
        let allMax = -Infinity, allMin = Infinity;
        for (let c = 0; c < nCh; c++) {
            for (let i = 0; i < traces[c].length; i++) {
                const v = traces[c][i];
                if (v > allMax) allMax = v;
                if (v < allMin) allMin = v;
            }
        }
        if (!isFinite(allMax) || !isFinite(allMin) || allMax === allMin) {
            allMax = 1; allMin = -1;
        }
        const targetMax = allMax * 1.2;
        const targetMin = allMin * 1.2;
        // EMA smoothing on y-limits so scale eases instead of jumps
        const smooth = 0.2;
        eegYLimits.max = eegYLimits.max * (1 - smooth) + targetMax * smooth;
        eegYLimits.min = eegYLimits.min * (1 - smooth) + targetMin * smooth;
        const yMin = eegYLimits.min;
        const yMax = eegYLimits.max;

        // Plot area inside frame with margins for labels.
        const padL = 46, padR = 6, padT = 14, padB = 22;
        const plotX = x0 + padL, plotY = y0 + padT;
        const plotW = w - padL - padR, plotH = h - padT - padB;

        // Y-axis ticks (3): min, 0, max (or simply min/mid/max if 0 is out of range)
        ctx.fillStyle = '#888';
        ctx.font = '10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const yTicks = [yMax, (yMax + yMin) / 2, yMin];
        for (const tv of yTicks) {
            const py = plotY + plotH * (1 - (tv - yMin) / (yMax - yMin));
            ctx.fillText(tv.toFixed(0), plotX - 4, py);
            ctx.strokeStyle = '#222';
            ctx.beginPath(); ctx.moveTo(plotX, py); ctx.lineTo(plotX + plotW, py); ctx.stroke();
        }
        // Y-axis label
        ctx.save();
        ctx.translate(x0 + 12, y0 + h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillStyle = '#aaa';
        ctx.fillText('μV (filtered)', 0, 0);
        ctx.restore();

        // X-axis ticks: -1.5s, -1.0s, -0.5s, 0
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = '#888';
        const nXTicks = 4;
        for (let i = 0; i < nXTicks; i++) {
            const frac = i / (nXTicks - 1);
            const tval = -windowS * (1 - frac);
            const px = plotX + plotW * frac;
            ctx.fillText(tval.toFixed(1), px, plotY + plotH + 4);
            ctx.strokeStyle = '#222';
            ctx.beginPath(); ctx.moveTo(px, plotY); ctx.lineTo(px, plotY + plotH); ctx.stroke();
        }
        ctx.fillStyle = '#aaa';
        ctx.fillText('time (s, 0 = now)', x0 + w / 2, y0 + h - 11);

        // Channel colors (4 channels, distinct hues)
        const colors = ['#33ff66', '#ffe600', '#33ccff', '#ff5252'];
        for (let c = 0; c < nCh; c++) {
            const trace = traces[c];
            if (!trace.length) continue;
            ctx.strokeStyle = colors[c % colors.length];
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < trace.length; i++) {
                const fx = i / (trace.length - 1);
                const px = plotX + plotW * fx;
                const v = trace[i];
                const py = plotY + plotH * (1 - (v - yMin) / (yMax - yMin));
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        // Per-channel legend (top-right corner of plot area)
        ctx.font = '10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        const idxs = live.channel_indices || [];
        for (let c = 0; c < nCh; c++) {
            ctx.fillStyle = colors[c % colors.length];
            const label = (idxs[c] != null ? 'ch' + idxs[c] : 'ch' + c)
                + (live.channels && live.channels[c] ? ' σ=' + live.channels[c].std.toFixed(0) : '');
            ctx.fillText(label, x0 + w - 6, y0 + 4 + c * 12);
        }
        ctx.restore();
    }

    function drawEegBadge(ctx, W, H, live) {
        // Compact "is the headset streaming?" indicator. Drawn in top-right of
        // the center play area during TRIAL/REST states so the user can see at
        // a glance that data is still flowing without having to read the full
        // readout.
        const bandW = W * BAND_FRACTION;
        const x = W - bandW - 12;
        const y = 16;
        ctx.font = '12px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        if (!live) {
            ctx.fillStyle = '#666';
            ctx.fillText('● EEG ?', x, y);
        } else if (!live.streaming) {
            ctx.fillStyle = COL_WARN;
            ctx.fillText('● EEG ✗', x, y);
        } else {
            ctx.fillStyle = '#33ff66';
            ctx.fillText('● EEG · ' + (live.n_samples || 0), x, y);
        }
    }

    function drawRecordCueLayer(ctx, W, H, status, nowMs) {
        const bandW = W * BAND_FRACTION;
        const centerX = W / 2;
        const centerY = H / 2;
        const rec = dashState.record || {};
        const recState = rec.recState || {};
        const trialDur = rec.trialDurationS || 15;
        const restDur  = rec.restDurationS  || 3;
        const totalTrials = rec.totalTrials || 40;

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = COL_SCORE;

        if (status === 'RECORD_INIT') {
            ctx.font = '24px ui-monospace, Menlo, monospace';
            ctx.fillText('initializing — measuring display refresh…', centerX, centerY);
            return;
        }
        if (status === 'RECORD_READY') {
            // Top: prompt + flicker freq readout
            ctx.font = 'bold 24px ui-monospace, Menlo, monospace';
            ctx.fillText('press SPACE to begin', centerX, 52);
            ctx.font = '12px ui-monospace, Menlo, monospace';
            ctx.fillStyle = '#888';
            const m = window.dash_clientside && window.dash_clientside.brainpong_measurement;
            if (m && m.ok) {
                ctx.fillText(
                    'refresh ' + (m.measuredHz || 0).toFixed(1) + ' Hz · stimulus '
                    + (m.actualLeftHz || 0).toFixed(2) + ' / ' + (m.actualRightHz || 0).toFixed(2) + ' Hz',
                    centerX, 76);
            } else if (m) {
                ctx.fillStyle = COL_WARN;
                ctx.fillText('refresh measurement not OK — flicker may be wrong', centerX, 76);
            }

            // Middle: live filtered EEG plot in the center play area
            const bandW = W * BAND_FRACTION;
            const plotW = (W - 2 * bandW) - 24;
            const plotH = 240;
            const plotX = bandW + 12;
            const plotY = (H - plotH) / 2;
            drawEegPlot(ctx, plotX, plotY, plotW, plotH, rec.eegLive);

            // Status line below the plot
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.font = '12px ui-monospace, Menlo, monospace';
            const live = rec.eegLive;
            if (!live) {
                ctx.fillStyle = '#888';
                ctx.fillText('EEG: waiting for first sample…', centerX, plotY + plotH + 12);
            } else if (!live.streaming) {
                ctx.fillStyle = COL_WARN;
                ctx.fillText('✗ EEG NOT streaming — ' + (live.reason || 'unknown'),
                             centerX, plotY + plotH + 12);
            } else {
                ctx.fillStyle = '#33ff66';
                ctx.fillText('✓ EEG · ' + (live.srate || '?') + ' Hz · '
                    + (live.n_samples || 0) + ' raw samples · plot @ '
                    + (live.effective_hz || '?') + ' Hz over '
                    + (live.window_s || 0).toFixed(1) + 's',
                    centerX, plotY + plotH + 12);
            }
            return;
        }
        if (status === 'RECORD_TRIAL') {
            const side = recState.side || 'L';
            const arrow = side === 'L' ? '←' : '→';
            const trialIdx = (recState.trial_idx != null ? recState.trial_idx : 0) + 1;

            // Trial counter
            ctx.font = '14px ui-monospace, Menlo, monospace';
            ctx.fillStyle = '#888';
            ctx.textBaseline = 'top';
            ctx.fillText('Trial ' + trialIdx + ' / ' + totalTrials, centerX, 24);

            // Big arrow
            ctx.font = 'bold 160px ui-monospace, Menlo, monospace';
            ctx.fillStyle = COL_SCORE;
            ctx.textBaseline = 'middle';
            ctx.fillText(arrow, centerX, centerY);

            // Countdown bar (anchored to the moment we observed RECORD_TRIAL)
            if (trialPhaseStartMs !== null) {
                const elapsed = (nowMs - trialPhaseStartMs) / 1000;
                const remaining = Math.max(0, trialDur - elapsed);
                const fracLeft = remaining / trialDur;
                const barW = (W - 2 * bandW) * 0.7;
                const barH = 8;
                const barX = centerX - barW / 2;
                const barY = H - 70;
                ctx.fillStyle = '#333';
                ctx.fillRect(barX, barY, barW, barH);
                ctx.fillStyle = '#33ff66';
                ctx.fillRect(barX, barY, barW * fracLeft, barH);
            }

            // Hold indicator (if space is currently down)
            if (window.dash_clientside && window.dash_clientside.brainpong_space_held) {
                ctx.font = '16px ui-monospace, Menlo, monospace';
                ctx.fillStyle = '#33ff66';
                ctx.textBaseline = 'top';
                ctx.fillText('● HOLDING', centerX, 50);
            }
            // Small live-EEG indicator (top-right of center area)
            drawEegBadge(ctx, W, H, rec.eegLive);
            return;
        }
        if (status === 'RECORD_REST') {
            ctx.font = 'bold 64px ui-monospace, Menlo, monospace';
            ctx.fillStyle = '#888';
            ctx.fillText('+', centerX, centerY);
            drawEegBadge(ctx, W, H, rec.eegLive);
            return;
        }
        if (status === 'RECORD_DONE') {
            ctx.font = 'bold 28px ui-monospace, Menlo, monospace';
            ctx.fillText('saving…', centerX, centerY);
            return;
        }
        if (status === 'RECORD_SAVED') {
            ctx.font = 'bold 28px ui-monospace, Menlo, monospace';
            ctx.fillText('DONE', centerX, centerY - 18);
            const ss = rec.saveStatus || {};
            ctx.font = '14px ui-monospace, Menlo, monospace';
            ctx.fillStyle = ss.error ? COL_WARN : '#888';
            const line = ss.error
                ? ('error: ' + ss.error)
                : ('saved: ' + (ss.saved_path || '(unknown path)'));
            ctx.fillText(line, centerX, centerY + 12);
            return;
        }
    }

    function draw(ctx, W, H, isLeftOn, isRightOn, nowMs) {
        const bandW = W * BAND_FRACTION;
        const rightBandX = W * (1 - BAND_FRACTION);
        const status = (dashState.appStatus && dashState.appStatus.status) || 'STARTING';
        const recordingStatus = isRecordingStatus(status);

        ctx.fillStyle = COL_BG_CENTER;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = isLeftOn ? COL_FLICKER_ON : COL_FLICKER_OFF;
        ctx.fillRect(0, 0, bandW, H);
        ctx.fillStyle = isRightOn ? COL_FLICKER_ON : COL_FLICKER_OFF;
        ctx.fillRect(rightBandX, 0, bandW, H);

        if (recordingStatus) {
            drawRecordCueLayer(ctx, W, H, status, nowMs);
        } else {
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

            // Anchor countdown bars to the moment a phase began (independent of
            // server-side 500 ms tick granularity).
            if (status !== lastObservedStatus) {
                if (status === 'RECORD_TRIAL') trialPhaseStartMs = nowMs;
                else if (status === 'RECORD_REST') restPhaseStartMs = nowMs;
                lastObservedStatus = status;
            }

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

            draw(ctx, W, H, isLeftOn, isRightOn, nowMs);
        }

        frameIdx++;
        requestAnimationFrame(loop);
    }

    // ---------------- Public entry point ----------------
    // Dash clientside_callback calls this whenever any of the input stores
    // change. First call boots the rAF loop; subsequent calls just refresh
    // the state mirror.
    function renderPong(canvasId, gameState, appStatus, settings, noBoard, freqLeft, freqRight, record) {
        dashState.canvasId  = canvasId;
        dashState.gameState = gameState;
        dashState.appStatus = appStatus;
        dashState.settings  = settings;
        dashState.noBoard   = !!noBoard;
        dashState.freqLeft  = freqLeft;
        dashState.freqRight = freqRight;
        dashState.record    = record || null;

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
