// BrainPong — EOG clientside renderer.
// requestAnimationFrame loop owns all drawing; Dash pushes state into a
// closure-scoped mirror on every callback fire.

if (!window.dash_clientside) { window.dash_clientside = {}; }

(function () {
    "use strict";

    const PADDLE_HEIGHT = 20;
    const BALL_RADIUS   = 10;

    const COL_BG            = '#1a1a1a';
    const COL_PADDLE_AI     = '#ff5252';
    const COL_PADDLE_PLAYER = '#33ff66';
    const COL_BALL          = '#ffe600';
    const COL_SCORE         = '#FFFFFF';

    const FREQ_LEFT     = 880.00;    // A5  — player left
    const FREQ_RIGHT    = 987.77;    // B5  — player right
    const FREQ_AI_LEFT  = 1174.66;   // D6  — AI left
    const FREQ_AI_RIGHT = 1318.51;   // E6  — AI right

    const dashState = {
        gameState: null,
        appStatus: null,
        settings:  null,
        canvasId:  null,
    };

    let started         = false;
    let prevZoneIdx     = null;
    let prevAiX         = null;
    let audioCtx        = null;

    function getAudioCtx() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            document.addEventListener('click', function () {
                if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
            });
        }
        return audioCtx;
    }

    function playTone(freq) {
        const ctx = getAudioCtx();
        if (ctx.state === 'suspended') ctx.resume().catch(function () {});
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type            = 'sine';
        osc.frequency.value = freq;
        const t = ctx.currentTime;
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(0.35, t + 0.01);
        gain.gain.linearRampToValueAtTime(0, t + 0.09);
        osc.start(t);
        osc.stop(t + 0.09);
    }

    function draw(ctx, W, H) {
        ctx.fillStyle = COL_BG;
        ctx.fillRect(0, 0, W, H);

        const gs = dashState.gameState;
        if (!gs) return;

        const paddleW = (dashState.settings && dashState.settings.paddle_width) || 264;

        // Ghost outlines at all three zone positions
        const ghostX = [W / 6, W / 2, 5 * W / 6];
        ctx.fillStyle = 'rgba(180, 180, 180, 0.12)';
        for (const gx of ghostX) {
            ctx.fillRect(gx - paddleW / 2, 0,               paddleW, PADDLE_HEIGHT);
            ctx.fillRect(gx - paddleW / 2, H - PADDLE_HEIGHT, paddleW, PADDLE_HEIGHT);
        }

        // Active paddles
        ctx.fillStyle = COL_PADDLE_AI;
        ctx.fillRect(gs.ai_x     - paddleW / 2, 0,               paddleW, PADDLE_HEIGHT);
        ctx.fillStyle = COL_PADDLE_PLAYER;
        ctx.fillRect(gs.player_x - paddleW / 2, H - PADDLE_HEIGHT, paddleW, PADDLE_HEIGHT);

        // Ball
        ctx.fillStyle = COL_BALL;
        ctx.beginPath();
        ctx.arc(gs.ball_x, gs.ball_y, BALL_RADIUS, 0, 2 * Math.PI);
        ctx.fill();

        // Score
        ctx.font         = 'bold 40px ui-monospace, Menlo, monospace';
        ctx.textAlign    = 'center';
        ctx.fillStyle    = COL_SCORE;
        ctx.textBaseline = 'top';
        ctx.fillText(String(gs.ai_score),     W / 2, 12);
        ctx.textBaseline = 'bottom';
        ctx.fillText(String(gs.player_score), W / 2, H - PADDLE_HEIGHT - 8);
    }

    function loop() {
        const canvas = dashState.canvasId ? document.getElementById(dashState.canvasId) : null;
        if (canvas) {
            draw(canvas.getContext('2d'), canvas.width, canvas.height);
        }
        requestAnimationFrame(loop);
    }

    function renderPong(canvasId, gameState, appStatus, settings) {
        dashState.canvasId  = canvasId;
        dashState.appStatus = appStatus;
        dashState.settings  = settings;

        const playing = appStatus && appStatus.status === 'PLAYING';
        if (gameState && playing) {
            const z = gameState.zone_idx;
            if (prevZoneIdx !== null && z !== prevZoneIdx) {
                playTone(z < prevZoneIdx ? FREQ_LEFT : FREQ_RIGHT);
            }
            prevZoneIdx = z;

            const ax = gameState.ai_x;
            if (prevAiX !== null && ax !== prevAiX) {
                playTone(ax < prevAiX ? FREQ_AI_LEFT : FREQ_AI_RIGHT);
            }
            prevAiX = ax;

        } else {
            prevZoneIdx = null;
            prevAiX     = null;
        }

        dashState.gameState = gameState;
        if (!started) {
            started = true;
            requestAnimationFrame(loop);
        }
    }

    window.dash_clientside.renderPong = renderPong;
})();
