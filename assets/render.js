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

    const WAVE_YMAX = 40000;   // fixed ±40k µV axis

    const dashState = { gameState: null, appStatus: null, settings: null, canvasId: null };
    let started = false, prevZoneIdx = null, prevAiX = null, audioCtx = null, trails = [];

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
        } else if (type === 'ice') {
            ctx.fillStyle = '#3388ff'; ctx.beginPath(); ctx.arc(x, y, POWERUP_RADIUS, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = '#fff'; ctx.font = '18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('❄', x, y);
        } else {
            ctx.fillStyle = '#aa33ff'; ctx.beginPath(); ctx.arc(x, y, POWERUP_RADIUS, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = '#fff';
            for (const o of [{ dx: 0, dy: -6 }, { dx: -5, dy: 4 }, { dx: 5, dy: 4 }]) { ctx.beginPath(); ctx.arc(x + o.dx, y + o.dy, 3, 0, 2 * Math.PI); ctx.fill(); }
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
        // balls (white + glow)
        ctx.save(); ctx.shadowColor = COL_BALL; ctx.shadowBlur = 18; ctx.fillStyle = COL_BALL;
        for (const b of (gs.balls || [])) { ctx.beginPath(); ctx.arc(b.x, flipY(b.y), BALL_RADIUS, 0, 2 * Math.PI); ctx.fill(); }
        ctx.restore();
    }
    function recordTrails() {
        const gs = dashState.gameState;
        const playing = dashState.appStatus && dashState.appStatus.status === 'PLAYING';
        if (!gs || !playing) { trails = []; return; }
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
        if (t) { t.textContent = title; t.style.color = color || COL_INK; }
        if (s) s.innerHTML = sub || '';
        if (h) h.textContent = hint || '';
    }
    function updateOverlay(appStatus, gameState) {
        const stage = document.getElementById('game-stage');
        const status = (appStatus && appStatus.status) || 'STARTING';
        const cd = Math.max(0, Math.ceil((appStatus && appStatus.countdown) || 0));
        if (status === 'PLAYING') { if (stage) stage.classList.remove('frozen'); return; }
        if (status === 'STARTING') {
            setOverlay('BrainPong', '2-player EOG · glance to move', 'press ↻ to start', COL_INK);
        } else if (status === 'INSTRUCTIONS') {
            setOverlay('Get Ready', 'Hands off the keyboard', 'calibrating in ' + cd + 's', COL_P1);
        } else if (status === 'CALIBRATING') {
            setOverlay('Calibration', 'Both players — look at the center of the screen<br>Hold still · don’t blink · no talking',
                       'measuring baseline — ' + cd + 's', COL_P2);
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
        const playing = appStatus && appStatus.status === 'PLAYING';
        if (gameState && playing) {
            const z = gameState.zone_idx;
            if (prevZoneIdx !== null && z !== prevZoneIdx) playTone(z < prevZoneIdx ? FREQ_LEFT : FREQ_RIGHT);
            prevZoneIdx = z;
            const ax = gameState.ai_x;
            if (prevAiX !== null && ax !== prevAiX) playTone(ax < prevAiX ? FREQ_AI_LEFT : FREQ_AI_RIGHT);
            prevAiX = ax;
        } else { prevZoneIdx = null; prevAiX = null; }
        dashState.gameState = gameState;
        if (!started) { started = true; requestAnimationFrame(fieldLoop); }
    }

    // ==========================================================
    //  WAVEFORM  (fixed ±40k µV, dashed ±Nσ, second gridlines, soft
    //  peak-centered glance bands: green = LEFT, red = RIGHT).
    //  waveData = { y:[...velocity samples...], thr, sigma_thr, win_s }
    // ==========================================================
    function renderWave(canvasId, waveData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        const { ctx, w, h } = fit(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.font = '9px ui-monospace, monospace';
        const yOf = v => h / 2 - (Math.max(-WAVE_YMAX, Math.min(WAVE_YMAX, v)) / WAVE_YMAX) * (h / 2 - 8);
        const y = (waveData && waveData.y) || [];
        const n = y.length;
        const xOf = i => (n > 1 ? i / (n - 1) : 0) * w;
        const thr = waveData && waveData.thr;
        const winS = (waveData && waveData.win_s) || 5;

        // soft peak-centered glance bands (behind the trace)
        if (thr && n) {
            const minRun = 2, L = COL_LEFT, R = COL_RIGHT;
            let i = 0;
            while (i < n) {
                const sgn = y[i] > thr ? 1 : (y[i] < -thr ? -1 : 0);
                if (!sgn) { i++; continue; }
                let j = i, peak = i, pv = Math.abs(y[i]);
                while (j < n && Math.sign(y[j]) === sgn && Math.abs(y[j]) > thr * 0.5) { if (Math.abs(y[j]) > pv) { pv = Math.abs(y[j]); peak = j; } j++; }
                if (j - i >= minRun) {
                    const col = sgn < 0 ? L : R, cx = xOf(peak), half = Math.max((xOf(j) - xOf(i)) * 0.9, 10);
                    const x0 = cx - half, x1 = cx + half;
                    const g = ctx.createLinearGradient(x0, 0, x1, 0);
                    g.addColorStop(0, 'transparent'); g.addColorStop(.5, col); g.addColorStop(1, 'transparent');
                    ctx.globalAlpha = .45; ctx.fillStyle = g; ctx.fillRect(x0, 0, x1 - x0, h); ctx.globalAlpha = 1;
                }
                i = j;
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
        // dashed sigma lines
        if (thr) {
            ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = COL_SIGMA;
            ctx.beginPath(); ctx.moveTo(0, yOf(thr)); ctx.lineTo(w, yOf(thr)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, yOf(-thr)); ctx.lineTo(w, yOf(-thr)); ctx.stroke(); ctx.restore();
            const st = (waveData && waveData.sigma_thr) || '';
            ctx.fillStyle = COL_SIGMA; ctx.textBaseline = 'alphabetic';
            ctx.fillText('+' + st + 'σ', 4, yOf(thr) - 3); ctx.fillText('-' + st + 'σ', 4, yOf(-thr) + 10);
        }
        // axis labels (µV on the left, with the ±40k)
        ctx.fillStyle = COL_INK_DIM; ctx.textBaseline = 'top'; ctx.fillText('+40k µV', 4, 3);
        ctx.textBaseline = 'bottom'; ctx.fillText('-40k µV', 4, h - 12);
        // trace
        if (n) {
            ctx.lineJoin = 'round'; ctx.lineWidth = 1.6; ctx.strokeStyle = COL_TRACE;
            ctx.beginPath();
            for (let k = 0; k < n; k++) { const x = xOf(k), yy = yOf(y[k]); k ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy); }
            ctx.stroke();
        }
        return null;
    }

    // ==========================================================
    //  RESPONSIVE LAYOUT — size the field to the biggest square that fits, and pin
    //  the grid height to that side so the left / center / right boxes line up.
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
        const side = Math.max(80, Math.min(availH, availW - leftCol.offsetWidth - rightCol.offsetWidth - 2 * gap));
        grid.style.height   = Math.floor(side) + 'px';
        cabinet.style.width = Math.floor(side) + 'px';
        cabinet.style.height = Math.floor(side) + 'px';
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
