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

    const dashState = {
        gameState: null,
        appStatus: null,
        settings:  null,
        canvasId:  null,
    };

    let started = false;

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
        dashState.gameState = gameState;
        dashState.appStatus = appStatus;
        dashState.settings  = settings;
        if (!started) {
            started = true;
            requestAnimationFrame(loop);
        }
    }

    window.dash_clientside.renderPong = renderPong;
})();
