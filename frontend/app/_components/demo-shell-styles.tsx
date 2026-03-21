'use client';

export default function DemoShellStyles() {
  return (
    <style jsx global>{`
      .demo-shell,
      .demo-shell *,
      .demo-shell *::before,
      .demo-shell *::after {
        box-sizing: border-box;
      }

      .demo-shell {
        --demo-card-bg: rgba(7, 17, 29, 0.84);
        --demo-card-border: rgba(255, 255, 255, 0.08);
        --demo-panel-bg:
          linear-gradient(180deg, rgba(20, 34, 55, 0.9), rgba(10, 18, 30, 0.9));
        --demo-surface-bg:
          radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 24%),
          linear-gradient(180deg, #07111d 0%, #0d1624 52%, #152033 100%);
        min-height: 100vh;
        padding: 32px;
        background:
          radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
          radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.12), transparent 24%),
          linear-gradient(160deg, #07111d 0%, #0e1726 48%, #1b2638 100%);
        color: #f8fafc;
        font-family: var(--font-geist-sans), "Avenir Next", "Segoe UI", sans-serif;
      }

      .demo-shell .hero-card,
      .demo-shell .canvas-card,
      .demo-shell .metric-card {
        border: 1px solid var(--demo-card-border);
        background: var(--demo-card-bg);
        backdrop-filter: blur(18px);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
      }

      .demo-shell .hero-card {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.9fr);
        gap: 24px;
        padding: 28px;
        border-radius: 28px;
        margin-bottom: 24px;
      }

      .demo-shell .hero-copy h1,
      .demo-shell .section-head h2 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 600;
        letter-spacing: -0.03em;
      }

      .demo-shell .hero-copy h1 {
        font-size: clamp(2.3rem, 4vw, 4.3rem);
        line-height: 0.94;
        max-width: 12ch;
      }

      .demo-shell .eyebrow,
      .demo-shell .mini-label {
        margin: 0 0 10px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: rgba(226, 232, 240, 0.72);
      }

      .demo-shell .lede,
      .demo-shell .annotation,
      .demo-shell .detail-copy,
      .demo-shell .notes-list,
      .demo-shell .legend-strip,
      .demo-shell .metric-subcopy,
      .demo-shell .shell-copy {
        color: rgba(226, 232, 240, 0.82);
      }

      .demo-shell .lede {
        max-width: 64ch;
        line-height: 1.62;
        margin: 18px 0 0;
        font-size: 1rem;
      }

      .demo-shell .detail-copy {
        margin: 0;
        line-height: 1.62;
      }

      .demo-shell .detail-copy.subtle,
      .demo-shell .metric-subcopy {
        color: rgba(226, 232, 240, 0.68);
      }

      .demo-shell .control-panel {
        display: flex;
        flex-direction: column;
        gap: 20px;
        justify-content: space-between;
        padding: 18px;
        border-radius: 24px;
        background: var(--demo-panel-bg);
      }

      .demo-shell .button-row,
      .demo-shell .status-strip,
      .demo-shell .legend-strip {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }

      .demo-shell .control-panel button {
        border-radius: 999px;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
      }

      .demo-shell .control-panel select,
      .demo-shell .control-panel input[type='number'] {
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(2, 6, 23, 0.68);
        color: #f8fafc;
      }

      .demo-shell .content-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.75fr);
        gap: 24px;
        align-items: start;
      }

      .demo-shell .sidebar {
        display: grid;
        gap: 24px;
      }

      .demo-shell .canvas-card,
      .demo-shell .metric-card {
        border-radius: 24px;
      }

      .demo-shell .canvas-card,
      .demo-shell .metric-card {
        padding: 18px;
      }

      .demo-shell .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
      }

      .demo-shell .section-head.compact {
        margin-bottom: 14px;
      }

      .demo-shell .annotation {
        margin: 0;
        max-width: 36ch;
        font-size: 0.92rem;
        line-height: 1.45;
        text-align: right;
      }

      .demo-shell .viz-surface {
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: var(--demo-surface-bg);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
      }

      .demo-shell .viz-surface canvas,
      .demo-shell .viz-surface svg {
        display: block;
        width: 100%;
        height: auto;
      }

      .demo-shell .surface-inset {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(2, 6, 23, 0.55);
      }

      .demo-shell .metric-grid-shell {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .demo-shell .notes-list {
        margin: 0;
        padding-left: 18px;
        line-height: 1.62;
      }

      .demo-shell code {
        font-family: var(--font-geist-mono), "SFMono-Regular", "Menlo", monospace;
        background: rgba(255, 255, 255, 0.08);
        padding: 2px 6px;
        border-radius: 8px;
      }

      @media (max-width: 1180px) {
        .demo-shell .hero-card,
        .demo-shell .content-grid {
          grid-template-columns: 1fr;
        }

        .demo-shell .annotation {
          text-align: left;
          max-width: none;
        }
      }

      @media (max-width: 720px) {
        .demo-shell {
          padding: 18px;
        }

        .demo-shell .hero-card,
        .demo-shell .canvas-card,
        .demo-shell .metric-card {
          border-radius: 22px;
        }

        .demo-shell .hero-card {
          padding: 20px;
        }

        .demo-shell .control-panel {
          padding: 16px;
        }

        .demo-shell .metric-grid-shell {
          grid-template-columns: 1fr;
        }
      }
    `}</style>
  );
}
