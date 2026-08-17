// BrainPong — EOG clientside renderer (Balatro-forward UI revamp).
// PRESENTATION ONLY. Draws the game; never touches physics, detection,
// recording, or any store contract. The play field is drawn VERTICALLY
// FLIPPED (draw-y = H - physics-y) so Player 1 (player_x, purple) is on TOP
// and Player 2 / AI (ai_x, yellow) on the BOTTOM — zero change to physics.
//
// Exposes two entry points used by clientside callbacks in the game script:
//   window.dash_clientside.renderPong(canvasId, gameState, appStatus, settings)
//   window.dash_clientside.renderWave(canvasId, waveData)

if (!window.dash_clientside) { window.dash_clientside = {}; }

(function () {
    "use strict";

    // ---- config mirrored from pong_game_brainflow.py (presentation copies) ----
    const PADDLE_HEIGHT  = 20;
    const BALL_RADIUS    = 10;
    const POWERUP_RADIUS = 14;
    const N_PANELS       = 5;
    const GHOST_GAP      = 12;   // px gap between adjacent paddle-slot boxes (visual only)

    const COL_SCREEN     = '#0b151c';
    const COL_P1         = '#b06bff', COL_P1_DEEP = '#5f2fb8';   // John  — TOP
    const COL_P2         = '#ffcf33', COL_P2_DEEP = '#b07f00';   // Esther/AI — BOTTOM
    const COL_BALL       = '#f7f8fc';
    const COL_GHOST_FILL = 'rgba(236,220,174,.09)';
    const COL_GHOST_LINE = 'rgba(236,220,174,.32)';
    const COL_TRACE      = '#ecdcae';
    const COL_LEFT       = '#46e08a';   // LEFT glance band
    const COL_RIGHT      = '#ff5b57';   // RIGHT glance band
    const COL_SIGMA      = '#7d8fb8';
    const COL_INK        = '#f4ead2', COL_INK_DIM = '#90a4b0';

    const FREQ_LEFT = 880.00, FREQ_RIGHT = 987.77, FREQ_AI_LEFT = 1174.66, FREQ_AI_RIGHT = 1318.51;
    // serve-hold cues (distinct from the four move tones above):
    // READY/SET ticks E5, GO! C6; between-point ticks C5, launch G5.
    const FREQ_TICK_START = 659.25, FREQ_GO = 1046.50, FREQ_TICK_SERVE = 523.25, FREQ_LAUNCH = 783.99;

    const WAVE_YMAX_DEFAULT = 15000;   // ±µV waveform half-range (live slider: 5k–50k)

    const dashState = { gameState: null, appStatus: null, settings: null, canvasId: null };
    let started = false, prevZoneIdx = null, prevAiX = null, audioCtx = null, trails = [];
    let prevHold = null, goFlashUntil = 0;   // serve-hold audio bookkeeping + GO! flash

    // ---------------- audio ----------------
    function getAudioCtx() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const unlock = () => { if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume().catch(() => {}); };
            document.addEventListener('click', unlock);
            document.addEventListener('keydown', unlock);
            document.addEventListener('mousedown', unlock);
        }
        return audioCtx;
    }
    function playTone(freq) {
        const ctx = getAudioCtx();
        const go = () => {
            const osc = ctx.createOscillator(), gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.type = 'sine'; osc.frequency.value = freq;
            const t = ctx.currentTime;
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(0.35, t + 0.01);
            gain.gain.linearRampToValueAtTime(0, t + 0.09);
            osc.start(t); osc.stop(t + 0.09);
        };
        if (ctx.state === 'suspended') ctx.resume().then(go).catch(() => {}); else go();
    }

    // ---------------- shared canvas helpers ----------------
    function fit(canvas) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const r = canvas.getBoundingClientRect();
        const w = Math.max(2, Math.round(r.width)), h = Math.max(2, Math.round(r.height));
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
        const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx, w, h };
    }
    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath(); ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }

    // ==========================================================
    //  PLAY FIELD
    // ==========================================================
    function paddle(ctx, x, y, w, h, top, deep) {
        ctx.save(); ctx.shadowColor = top; ctx.shadowBlur = 16;
        const g = ctx.createLinearGradient(0, y, 0, y + h);
        g.addColorStop(0, top); g.addColorStop(1, deep);
        ctx.fillStyle = g; roundRect(ctx, x, y, w, h, 5); ctx.fill(); ctx.restore();
    }
    function drawPowerup(ctx, x, y, type) {
        ctx.save();
        if (type === 'fire') {
            ctx.fillStyle = '#ff3333'; ctx.beginPath(); ctx.arc(x, y, POWERUP_RADIUS, 0, 2 * Math.PI); ctx.fill();
            ctx.font = '18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('🔥', x, y);
        } else {
            ctx.fillStyle = '#aa33ff'; ctx.beginPath(); ctx.arc(x, y, POWERUP_RADIUS, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = '#fff';
            for (const o of [{ dx: -5, dy: 0 }, { dx: 5, dy: 0 }]) { ctx.beginPath(); ctx.arc(x + o.dx, y + o.dy, 3.5, 0, 2 * Math.PI); ctx.fill(); }
        }
        ctx.restore();
    }
    function drawField(ctx, W, H) {
        ctx.fillStyle = COL_SCREEN; ctx.fillRect(0, 0, W, H);
        const gs = dashState.gameState; if (!gs) return;
        // Slot boxes are one pitch wide minus a small gap so adjacent slots read as
        // separate spaces. The active paddle uses the SAME box, so it lines up exactly
        // with its ghost slot (physics collision width is unchanged; this is draw-only).
        const slotPitch = W / N_PANELS;
        const drawW = Math.max(20, slotPitch - GHOST_GAP);
        const flipY = y => H - y;

        // brighter ghost slots (both rows; flush to the edges so paddles align with them)
        for (let i = 0; i < N_PANELS; i++) {
            const gx = W * (2 * i + 1) / (2 * N_PANELS);
            ctx.fillStyle = COL_GHOST_FILL;
            roundRect(ctx, gx - drawW / 2, 0, drawW, PADDLE_HEIGHT, 6); ctx.fill();
            roundRect(ctx, gx - drawW / 2, H - PADDLE_HEIGHT, drawW, PADDLE_HEIGHT, 6); ctx.fill();
            ctx.lineWidth = 1.5; ctx.strokeStyle = COL_GHOST_LINE;
            roundRect(ctx, gx - drawW / 2, 0, drawW, PADDLE_HEIGHT, 6); ctx.stroke();
            roundRect(ctx, gx - drawW / 2, H - PADDLE_HEIGHT, drawW, PADDLE_HEIGHT, 6); ctx.stroke();
        }
        // ball trails (white, fading)
        for (const tr of trails) {
            for (let k = 0; k < tr.length; k++) {
                const p = tr[k], a = (k + 1) / tr.length;
                ctx.globalAlpha = a * 0.5; ctx.fillStyle = COL_BALL;
                ctx.beginPath(); ctx.arc(p.x, flipY(p.y), BALL_RADIUS * (0.3 + a * 0.65), 0, 2 * Math.PI); ctx.fill();
            }
        }
        ctx.globalAlpha = 1;
        // paddles — P1 (player_x, purple) flips to TOP (y=0); P2/AI (ai_x, yellow) flips to BOTTOM
        paddle(ctx, gs.player_x - drawW / 2, 0,                 drawW, PADDLE_HEIGHT, COL_P1, COL_P1_DEEP);
        paddle(ctx, gs.ai_x     - drawW / 2, H - PADDLE_HEIGHT, drawW, PADDLE_HEIGHT, COL_P2, COL_P2_DEEP);
        // powerups (flip position; icon upright)
        for (const pu of (gs.powerups || [])) drawPowerup(ctx, pu.x, flipY(pu.y), pu.type);
        // balls (white + glow) — hidden during the READY/SET/GO start hold so the
        // ball "appears" exactly on GO! (the between-point 'serve' hold keeps it
        // visible, parked at center: that stationary beat IS the round separator).
        if (!(gs.hold_kind === 'start' && (gs.serve_hold || 0) > 0)) {
            ctx.save(); ctx.shadowColor = COL_BALL; ctx.shadowBlur = 18; ctx.fillStyle = COL_BALL;
            for (const b of (gs.balls || [])) { ctx.beginPath(); ctx.arc(b.x, flipY(b.y), BALL_RADIUS, 0, 2 * Math.PI); ctx.fill(); }
            ctx.restore();
        }
        drawTrainingPrompts(ctx, W, H);
        drawServeCountdown(ctx, W, H);
    }

    // ---- READY / SET / GO! — the start-of-game countdown words (PLAYING only;
    //      training enters directly with no countdown). READY and SET each cover
    //      half the hold; GO! flashes AT the launch moment (goFlashUntil is armed
    //      by the audio bookkeeping in renderPong) and fades while the ball is
    //      already flying — racing-light semantics.
    function drawServeCountdown(ctx, W, H) {
        const gs = dashState.gameState, st = dashState.appStatus;
        if (!gs || !st || st.status !== 'PLAYING') return;
        const hold = gs.serve_hold || 0, total = gs.serve_hold_total || 0;
        let word = null;
        if (hold > 0 && total > 0 && gs.hold_kind === 'start') {
            word = hold > total / 2 ? { t: 'READY', c: COL_P1 } : { t: 'SET', c: COL_P2 };
        } else if (performance.now() < goFlashUntil) {
            word = { t: 'GO!', c: COL_INK };
        }
        if (!word) return;
        ctx.save();
        ctx.font = "96px 'm6x11', ui-monospace, monospace";
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.shadowColor = word.c; ctx.shadowBlur = 26; ctx.fillStyle = word.c;
        if (word.t === 'GO!') ctx.globalAlpha = Math.max(0, (goFlashUntil - performance.now()) / 450);
        ctx.fillText(word.t, W / 2, H / 2);
        ctx.restore();
    }

    // ---- TRAINING mode: per-player prompt lines on the field (never the dimming
    //      overlay — the field stays fully bright). P1's line sits in the upper half
    //      near the purple top paddle, P2's in the lower half near the yellow bottom
    //      paddle; each flips independently from gs.train_target / train_target_p2.
    //      P2 gets a prompt only when the bottom paddle is a human (settings.two_player).
    function drawTrainingPrompts(ctx, W, H) {
        const gs = dashState.gameState, st = dashState.appStatus;
        if (!gs || !st || st.status !== 'TRAINING') return;
        drawPromptLine(ctx, W, H * 0.32, gs.train_target || 'left', COL_P1);
        if (dashState.settings && dashState.settings.two_player) {
            drawPromptLine(ctx, W, H * 0.68, gs.train_target_p2 || 'left', COL_P2);
        }
        ctx.save();                                  // small mode watermark, mid-field
        ctx.globalAlpha = 0.55; ctx.fillStyle = COL_INK_DIM;
        ctx.font = "24px 'm6x11', ui-monospace, monospace";
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('TRAINING', W / 2, H / 2);
        ctx.restore();
    }
    function drawPromptLine(ctx, W, y, target, color) {
        const left  = target === 'left';
        const label = left ? 'MOVE ALL THE WAY LEFT' : 'MOVE ALL THE WAY RIGHT';
        const AW = 30, AH = 17, GAP = 20;            // arrow width/half-height, gap to text
        ctx.save();
        ctx.font = "36px 'm6x11', ui-monospace, monospace";
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.shadowColor = color; ctx.shadowBlur = 14; ctx.fillStyle = color;
        const tw = ctx.measureText(label).width;
        // arrow sits on the side it points to; the arrow+gap+text block stays centered
        const textCx = W / 2 + (left ? (AW + GAP) / 2 : -(AW + GAP) / 2);
        ctx.fillText(label, textCx, y);
        ctx.beginPath();
        if (left) {
            const ax = textCx - tw / 2 - GAP;        // arrow right edge
            ctx.moveTo(ax - AW, y); ctx.lineTo(ax, y - AH); ctx.lineTo(ax, y + AH);
        } else {
            const ax = textCx + tw / 2 + GAP;        // arrow left edge
            ctx.moveTo(ax + AW, y); ctx.lineTo(ax, y - AH); ctx.lineTo(ax, y + AH);
        }
        ctx.closePath(); ctx.fill();
        ctx.restore();
    }
    function recordTrails() {
        const gs = dashState.gameState;
        const playing = dashState.appStatus && dashState.appStatus.status === 'PLAYING';
        // no trail while the ball is parked for a serve hold (it would pile up as a
        // static glow at center — and the start-hold ball is hidden entirely)
        if (!gs || !playing || (gs.serve_hold || 0) > 0) { trails = []; return; }
        const balls = gs.balls || [];
        if (trails.length !== balls.length) trails = balls.map(() => []);
        for (let i = 0; i < balls.length; i++) { trails[i].push({ x: balls[i].x, y: balls[i].y }); if (trails[i].length > 16) trails[i].shift(); }
    }
    function fieldLoop() {
        const canvas = dashState.canvasId ? document.getElementById(dashState.canvasId) : null;
        if (canvas) { recordTrails(); drawField(canvas.getContext('2d'), canvas.width, canvas.height); }
        requestAnimationFrame(fieldLoop);
    }

    // ==========================================================
    //  SCOREBOARD + OVERLAY  (calibration / game over / paused / start)
    // ==========================================================
    function nameOf(id, fallback) { const el = document.getElementById(id); const v = el && el.value ? el.value.trim() : ''; return v || fallback; }
    function setOverlay(title, sub, hint, color) {
        const stage = document.getElementById('game-stage');
        const t = document.getElementById('overlay-title'), s = document.getElementById('overlay-sub'), h = document.getElementById('overlay-hint');
        if (stage) stage.classList.add('frozen');
        const ov = document.getElementById('game-overlay');
        if (ov) ov.classList.remove('calib');
        if (t) { t.textContent = title; t.style.color = color || COL_INK; }
        if (s) s.innerHTML = sub || '';
        if (h) h.textContent = hint || '';
    }
    function updateOverlay(appStatus, gameState) {
        const stage = document.getElementById('game-stage');
        const status = (appStatus && appStatus.status) || 'STARTING';
        const cd = Math.max(0, Math.ceil((appStatus && appStatus.countdown) || 0));
        // TRAINING renders like PLAYING: no overlay, no dim — the prompts live on the
        // field canvas (drawTrainingPrompts) so the whole game stays bright.
        if (status === 'PLAYING' || status === 'TRAINING') { if (stage) stage.classList.remove('frozen'); return; }
        if (status === 'STARTING') {
            setOverlay('BrainPong', '2-player EOG · glance to move', 'press ↻ to start · dumbbell to train', COL_INK);
        } else if (status === 'INSTRUCTIONS') {
            // All calibration instructions live here, before the dot appears — the
            // calibration scene itself must stay text-free so the eyes do not move.
            setOverlay('Get Ready', 'Hands off the keyboard<br>Look at the red dot · hold still<br>Don’t blink · no talking',
                       'calibrating in ' + cd + 's', COL_P1);
        } else if (status === 'CALIBRATING') {
            // No text — only the pulsing red fixation dot in the middle of the field.
            setOverlay('', '', '', COL_P2);
            const ov = document.getElementById('game-overlay');
            if (ov) ov.classList.add('calib');
        } else if (status === 'PAUSED') {
            setOverlay('Paused', '', 'press ‖ to resume', COL_INK);
        } else if (status === 'GAME_OVER') {
            const w = (appStatus && appStatus.winner) || '';
            const topWins = (w === 'P1' || w === 'Player');
            const who = topWins ? nameOf('p1-name', 'Player 1') : (w === 'AI' ? 'AI' : nameOf('p2-name', 'Player 2'));
            const color = topWins ? COL_P1 : COL_P2;
            const p1s = gameState ? (gameState.player_score || 0) : 0;
            const p2s = gameState ? (gameState.ai_score || 0) : 0;
            setOverlay(who + ' Wins', 'Final &nbsp;<span style="color:' + COL_P1 + '">' + p1s + '</span> : <span style="color:' + COL_P2 + '">' + p2s + '</span>',
                       'press ↻ for a new game', color);
        } else if (stage) { stage.classList.remove('frozen'); }
    }

    function renderPong(canvasId, gameState, appStatus, settings) {
        dashState.canvasId = canvasId; dashState.appStatus = appStatus; dashState.settings = settings;
        getAudioCtx();
        // scoreboard (P1/John = player_score = purple TOP; P2/Esther/AI = ai_score = yellow BOTTOM)
        if (gameState) {
            const p1 = document.getElementById('p1-score'), p2 = document.getElementById('p2-score');
            if (p1) p1.textContent = String(gameState.player_score || 0).padStart(2, '0');
            if (p2) p2.textContent = String(gameState.ai_score || 0).padStart(2, '0');
        }
        updateOverlay(appStatus, gameState);
        // movement tones play in TRAINING too — same audio feedback as a real rally
        const playing = appStatus && (appStatus.status === 'PLAYING' || appStatus.status === 'TRAINING');
        if (gameState && playing) {
            const z = gameState.zone_idx;
            if (prevZoneIdx !== null && z !== prevZoneIdx) playTone(z < prevZoneIdx ? FREQ_LEFT : FREQ_RIGHT);
            prevZoneIdx = z;
            const ax = gameState.ai_x;
            if (prevAiX !== null && ax !== prevAiX) playTone(ax < prevAiX ? FREQ_AI_LEFT : FREQ_AI_RIGHT);
            prevAiX = ax;
            // serve-hold cues (PLAYING only — training has no countdown): READY/SET
            // ticks + GO! note at game start; two low ticks + a launch tone around
            // every between-point hold. Edge-triggered off serve_hold crossings so
            // each cue fires exactly once.
            if (appStatus.status === 'PLAYING') {
                const hold = gameState.serve_hold || 0, total = gameState.serve_hold_total || 0;
                const kind = gameState.hold_kind;
                if (hold > 0 && total > 0) {
                    if (kind === 'start') {
                        if (prevHold === null || hold > prevHold) playTone(FREQ_TICK_START);            // READY
                        else if (prevHold > total / 2 && hold <= total / 2) playTone(FREQ_TICK_START);  // SET
                    } else if (prevHold !== null && hold <= prevHold) {
                        for (const b of [total * 2 / 3, total / 3]) {
                            if (prevHold > b && hold <= b) playTone(FREQ_TICK_SERVE);
                        }
                    }
                    prevHold = hold;
                } else if (prevHold !== null && prevHold > 0) {
                    playTone(kind === 'start' ? FREQ_GO : FREQ_LAUNCH);
                    if (kind === 'start') goFlashUntil = performance.now() + 450;   // GO! flash
                    prevHold = 0;
                }
            } else { prevHold = null; }
        } else { prevZoneIdx = null; prevAiX = null; prevHold = null; }
        dashState.gameState = gameState;
        if (!started) { started = true; requestAnimationFrame(fieldLoop); }
    }

    // ==========================================================
    //  WAVEFORM  — filtered GAZE AMPLITUDE (µV), auto-scaled, drawn as a min/max
    //  ENVELOPE band. Detection bands (green = LEFT, red = RIGHT) overlay the real
    //  motions the state machine counted. When the electrodes are railing the panel
    //  shows an explicit NO-CONTACT state instead of drawing garbage.
    //  waveData = { lo:[...µV...], hi:[...µV...], ymax, win_s, quality, fires }
    // ==========================================================
    function renderWave(canvasId, waveData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        const { ctx, w, h } = fit(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.font = '9px ui-monospace, monospace';
        const quality = (waveData && waveData.quality) || 'ok';
        const ymax = (waveData && waveData.ymax) || WAVE_YMAX_DEFAULT;
        const yOf = v => h / 2 - (Math.max(-ymax, Math.min(ymax, v)) / ymax) * (h / 2 - 8);
        const hi = (waveData && waveData.hi) || [];
        const lo = (waveData && waveData.lo) || [];
        const n = hi.length;
        const xOf = i => (n > 1 ? i / (n - 1) : 0) * w;
        const winS = (waveData && waveData.win_s) || 5;

        // NO-CONTACT state: raw electrodes railing → the trace is meaningless. Draw a
        // clear banner instead of a wall of clamped garbage, so the operator re-seats.
        if (quality === 'railed') {
            ctx.fillStyle = 'rgba(255,91,87,.10)'; ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = COL_RIGHT; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = COL_RIGHT; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.font = '700 12px ui-monospace, monospace';
            ctx.fillText('⚠ NO CONTACT', w / 2, h / 2 - 8);
            ctx.font = '9px ui-monospace, monospace';
            ctx.fillText('re-seat electrode', w / 2, h / 2 + 8);
            ctx.textAlign = 'left';
            return null;
        }

        // Detection bands — one per directional motion the glance-pair state machine
        // COUNTED (waveData.fires = [{x, dir}], x in [0,1] along the window): each
        // arming saccade and each return saccade, banded by its own detected
        // direction. green = LEFT, red = RIGHT.
        const fires = (waveData && waveData.fires) || [];
        if (fires.length) {
            const half = Math.max(w * 0.015, 8);
            for (const f of fires) {
                const cx = f.x * w, col = f.dir === 'LEFT' ? COL_LEFT : COL_RIGHT;
                ctx.globalAlpha = .32; ctx.fillStyle = col;
                ctx.fillRect(cx - half, 0, 2 * half, h); ctx.globalAlpha = 1;
            }
        }
        // second gridlines (static)
        ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.lineWidth = 1; ctx.fillStyle = COL_INK_DIM; ctx.textBaseline = 'bottom';
        for (let s = 0; s <= winS; s++) {
            const x = w - (s / winS) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            if (s > 0 && s < winS) ctx.fillText('-' + s + 's', x + 3, h - 2);
        }
        // zero line
        ctx.strokeStyle = 'rgba(255,255,255,.13)'; ctx.beginPath(); ctx.moveTo(0, yOf(0)); ctx.lineTo(w, yOf(0)); ctx.stroke();

        // axis labels — auto-scaled ±range, shown in µV or mV as appropriate
        const rngLbl = ymax >= 1000 ? ('±' + (ymax / 1000).toFixed(1) + ' mV')
                                    : ('±' + Math.round(ymax) + ' µV');
        ctx.fillStyle = COL_INK_DIM; ctx.textBaseline = 'top'; ctx.fillText(rngLbl, 4, 3);
        if (quality === 'calibrating') { ctx.fillStyle = COL_SIGMA; ctx.fillText('calibrating…', 4, 15); }

        // trace — filled min/max envelope of the gaze amplitude
        if (n) {
            ctx.beginPath();
            for (let k = 0; k < n; k++) { const x = xOf(k), yy = yOf(hi[k]); k ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); }
            for (let k = n - 1; k >= 0; k--) { ctx.lineTo(xOf(k), yOf(lo[k])); }
            ctx.closePath();
            ctx.globalAlpha = .30; ctx.fillStyle = COL_TRACE; ctx.fill(); ctx.globalAlpha = 1;
            ctx.lineJoin = 'round'; ctx.lineWidth = 1.4; ctx.strokeStyle = COL_TRACE;
            ctx.beginPath();
            for (let k = 0; k < n; k++) { const x = xOf(k), yy = yOf(hi[k]); k ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); }
            ctx.stroke();
            ctx.beginPath();
            for (let k = 0; k < n; k++) { const x = xOf(k), yy = yOf(lo[k]); k ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); }
            ctx.stroke();
        }
        return null;
    }

    // ==========================================================
    //  RESPONSIVE LAYOUT — every panel is a true SQUARE. The field gets the biggest
    //  square that fits while reserving MIN_W per side column; the side columns then
    //  split the leftover width, capped so their two square cards (plus the controls
    //  row on the left) still stack inside the field's height. Cards themselves are
    //  aspect-ratio 1 in CSS, so column width alone fixes their size.
    // ==========================================================
    function sizeLayout() {
        const stage = document.getElementById('game-stage');
        if (!stage) return;
        const grid    = stage.querySelector('.grid');
        const cabinet = document.getElementById('cabinet');
        const leftCol = stage.querySelector('.col.left');
        const rightCol = stage.querySelector('.col.right');
        if (!grid || !cabinet || !leftCol || !rightCol) return;
        const cs = getComputedStyle(stage);
        const availW = stage.clientWidth  - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
        const availH = stage.clientHeight - parseFloat(cs.paddingTop)  - parseFloat(cs.paddingBottom);
        const gap = parseFloat(getComputedStyle(grid).columnGap) || 24;
        const MIN_W = 170;                                   // width reserved per side column
        const controls = stage.querySelector('.controls');
        const ctrlH = (controls ? controls.offsetHeight : 128) + 24;   // + breathing room
        const side = Math.max(80, Math.min(availH, availW - 2 * MIN_W - 2 * gap));
        // Column width floor is LOWER than the reservation: in very short windows the
        // (side - ctrlH)/2 cap wins, and small squares beat overflowing the column.
        const w = Math.max(96, Math.min((availW - side - 2 * gap) / 2, (side - ctrlH) / 2));
        grid.style.height    = Math.floor(side) + 'px';
        cabinet.style.width  = Math.floor(side) + 'px';
        cabinet.style.height = Math.floor(side) + 'px';
        leftCol.style.width  = Math.floor(w) + 'px';
        rightCol.style.width = Math.floor(w) + 'px';
    }
    let _layoutObs = null;
    function initLayout() {
        const stage = document.getElementById('game-stage');
        if (!stage) { requestAnimationFrame(initLayout); return; }
        sizeLayout();
        if (window.ResizeObserver && !_layoutObs) { _layoutObs = new ResizeObserver(sizeLayout); _layoutObs.observe(stage); }
        window.addEventListener('resize', sizeLayout);
    }
    initLayout();

    // 'G' toggles the operator settings drawer (hidden from players by default).
    // Ignored while typing in the name fields so a name containing 'g' is fine.
    document.addEventListener('keydown', function (e) {
        if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
        if (e.key === 'g' || e.key === 'G') {
            const d = document.getElementById('tuning-drawer');
            if (d) d.style.display = (getComputedStyle(d).display === 'none') ? 'block' : 'none';
        }
    });

    window.dash_clientside.renderPong = renderPong;
    window.dash_clientside.renderWave = renderWave;
})();
