'use client';

import { useEffect, useRef, useState } from 'react';

type GridPoint = {
  i: number;
  j: number;
};

type Point = {
  x: number;
  y: number;
};

type VehiclePose = {
  x: number;
  y: number;
  yaw: number;
};

type CommandId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

type NearbySensorKey =
  | 'grassTopLeft'
  | 'grassTopCentre'
  | 'grassTopRight'
  | 'grassLeft'
  | 'grassRight'
  | 'grassBottomLeft'
  | 'grassBottomCentre'
  | 'grassBottomRight';

type GrassField = NearbySensorKey | 'grassCentre';

type SensorsSnapshot = {
  i: number;
  j: number;
  time: number;
  battery: number;
  rain: number;
  nearbyVisible: boolean;
  grassTopLeft: number;
  grassTopCentre: number;
  grassTopRight: number;
  grassLeft: number;
  grassCentre: number;
  grassRight: number;
  grassBottomLeft: number;
  grassBottomCentre: number;
  grassBottomRight: number;
};

type StatusSnapshot = {
  grassMean: number;
  grassMax: number;
};

type BehaviourDecision = {
  command: CommandId;
  title: string;
  detail: string;
  target: GridPoint;
};

type ActionRecord = {
  minute: number;
  command: CommandId;
  title: string;
  battery: number;
  rain: number;
};

type ActionAnimation = {
  decision: BehaviourDecision;
  sensors: SensorsSnapshot;
  fromCell: GridPoint;
  toCell: GridPoint;
  startPose: VehiclePose;
  endPose: VehiclePose;
  elapsed: number;
  duration: number;
};

type SimulationState = {
  grass: number[][];
  cell: GridPoint;
  pose: VehiclePose;
  battery: number;
  rain: number;
  time: number;
  previousCommand: CommandId;
  lastTravelCommand: CommandId;
  action: ActionAnimation | null;
  lastSensors: SensorsSnapshot;
  lastDecision: BehaviourDecision;
  status: StatusSnapshot;
  trail: Point[];
  recentActions: ActionRecord[];
  rngState: number;
  bladePhase: number;
  depleted: boolean;
};

type Telemetry = {
  minute: number;
  cellI: number;
  cellJ: number;
  battery: number;
  rain: number;
  grassMean: number;
  grassMax: number;
  commandLabel: string;
  behaviourTitle: string;
  behaviourDetail: string;
  sensorsVisible: boolean;
  target: GridPoint;
  actionProgress: number;
  depleted: boolean;
};

type RenderState = {
  telemetry: Telemetry;
  activeSensors: SensorsSnapshot;
  recentActions: ActionRecord[];
};

type SensorDisplay = {
  key: GrassField;
  label: string;
  di: number;
  dj: number;
};

const GRID_COLUMNS = 40;
const GRID_ROWS = 40;
const WALL_ROW = 20;
const WALL_END_COLUMN = 29;
const WALL_GUIDE_ROW = 19;
const WALL_OPENING_COLUMN = 30;
const CHARGING_STATION: GridPoint = { i: 0, j: 0 };

const ACTION_DURATION = 0.56;
const INITIAL_BATTERY = 0.92;
const INITIAL_RAIN = 0.001;
const CHARGE_RATE = 0.067;
const MOVE_DRAIN = 0.00385;
const IDLE_DRAIN = 0.0079;
const BASE_GROWTH = 0.00004;
const RAIN_GROWTH_FACTOR = 0.000015;
const PASS_CUT_FACTOR = 0.38;
const SOURCE_PASS_CUT_FACTOR = 0.65;
const MAX_TRAIL_POINTS = 480;
const MAX_HISTORY = 8;
const RANDOM_SEED = 0x5eede5;
const EPSILON = 1e-9;
const RETURN_SAFETY_MARGIN = 0.027;
const RAIN_AMOUNT_SCALE = 0.01;

const RAIN_LIMIT = 0.2;
const CHARGING_LEVEL = 1.0;
const LOW_BATTERY_LEVEL = 0.22;
const GRASS_CENTRE_LIMIT = 0.4;

const SENSOR_PRIORITY: Array<{
  key: NearbySensorKey;
  command: CommandId;
  threshold: number;
  detail: string;
}> = [
  {
    key: 'grassTopRight',
    command: 3,
    threshold: 0.4,
    detail: 'Top-right grass crossed its threshold, so the mower advances diagonally.',
  },
  {
    key: 'grassRight',
    command: 4,
    threshold: 0.1,
    detail: 'The right cell is promising, so the mower continues laterally.',
  },
  {
    key: 'grassBottomRight',
    command: 5,
    threshold: 0.4,
    detail: 'The bottom-right cell has enough growth to justify a diagonal move.',
  },
  {
    key: 'grassBottomCentre',
    command: 6,
    threshold: 0.001,
    detail: 'The cell below has any measurable grass, so the mower drops down.',
  },
  {
    key: 'grassBottomLeft',
    command: 7,
    threshold: 0.1,
    detail: 'The bottom-left cell clears the threshold, so the mower rotates clockwise into it.',
  },
  {
    key: 'grassLeft',
    command: 8,
    threshold: 0.5,
    detail: 'The left cell has strong growth and wins after the clockwise scan order.',
  },
  {
    key: 'grassTopLeft',
    command: 1,
    threshold: 0.1,
    detail: 'The top-left cell is worth visiting after the other candidates were checked first.',
  },
  {
    key: 'grassTopCentre',
    command: 2,
    threshold: 0.1,
    detail: 'The cell above is the final candidate in the clockwise sensor sweep.',
  },
];

const SENSOR_LAYOUT: SensorDisplay[] = [
  { key: 'grassTopLeft', label: 'Top-left', di: -1, dj: -1 },
  { key: 'grassTopCentre', label: 'Top', di: 0, dj: -1 },
  { key: 'grassTopRight', label: 'Top-right', di: 1, dj: -1 },
  { key: 'grassLeft', label: 'Left', di: -1, dj: 0 },
  { key: 'grassCentre', label: 'Centre', di: 0, dj: 0 },
  { key: 'grassRight', label: 'Right', di: 1, dj: 0 },
  { key: 'grassBottomLeft', label: 'Bottom-left', di: -1, dj: 1 },
  { key: 'grassBottomCentre', label: 'Bottom', di: 0, dj: 1 },
  { key: 'grassBottomRight', label: 'Bottom-right', di: 1, dj: 1 },
];

const COMMANDS: Record<
  CommandId,
  {
    label: string;
    short: string;
    di: number;
    dj: number;
  }
> = {
  0: { label: 'Stay / cut', short: 'STAY', di: 0, dj: 0 },
  1: { label: 'Up-left', short: 'UL', di: -1, dj: -1 },
  2: { label: 'Up', short: 'UP', di: 0, dj: -1 },
  3: { label: 'Up-right', short: 'UR', di: 1, dj: -1 },
  4: { label: 'Right', short: 'R', di: 1, dj: 0 },
  5: { label: 'Down-right', short: 'DR', di: 1, dj: 1 },
  6: { label: 'Down', short: 'D', di: 0, dj: 1 },
  7: { label: 'Down-left', short: 'DL', di: -1, dj: 1 },
  8: { label: 'Left', short: 'L', di: -1, dj: 0 },
};

const BEHAVIOUR_STACK = [
  '1. If the battery is not full and the mower is docked, stay on the charger.',
  '2. Else if the battery is low, follow the hand-coded return-home route around the wall opening.',
  '3. Else if grass under the mower is tall enough and rain is below 0.2, stay and cut it.',
  '4. Else if nearby sensor data exists, scan the neighbors in the original priority order, but snake diagonally into the next row at the map edges.',
  '5. Else if the mower reaches the wall guide row, move right to follow the wall and probe new areas.',
  '6. Else stay put to refresh the adjacent-cell sensor message for the next minute.',
];

export default function LawnmowerBBRPage() {
  const [initialSimulation] = useState(() => createSimulationState());
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simulationRef = useRef<SimulationState>(initialSimulation);

  const [running, setRunning] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(200);
  const [renderState, setRenderState] = useState<RenderState>(() =>
    createRenderState(initialSimulation, true),
  );
  const { telemetry, activeSensors, recentActions } = renderState;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }

    let frameId = 0;
    let lastTimestamp = performance.now();
    let uiCarry = 0;

    const loop = (timestamp: number) => {
      const rawDt = Math.min((timestamp - lastTimestamp) / 1000, 0.05);
      lastTimestamp = timestamp;

      if (running) {
        stepSimulation(simulationRef.current, rawDt * playbackSpeed);
        uiCarry += rawDt * playbackSpeed;

        const current = simulationRef.current;
        if (current.depleted) {
          setRunning(false);
          setRenderState(createRenderState(current, false));
        } else if (uiCarry >= 0.08) {
          uiCarry = 0;
          setRenderState(createRenderState(current, true));
        }
      }

      drawScene(context, canvas, simulationRef.current);
      frameId = window.requestAnimationFrame(loop);
    };

    drawScene(context, canvas, simulationRef.current);
    frameId = window.requestAnimationFrame(loop);

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [running, playbackSpeed]);

  const resetSimulation = () => {
    simulationRef.current = createSimulationState();
    setRunning(false);
    setRenderState(createRenderState(simulationRef.current, false));
  };

  const startSimulation = () => {
    if (simulationRef.current.depleted) {
      simulationRef.current = createSimulationState();
    }
    setRenderState(createRenderState(simulationRef.current, true));
    setRunning(true);
  };

  const advanceOneMinute = () => {
    runSingleMinute(simulationRef.current);
    setRenderState(createRenderState(simulationRef.current, false));
  };

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Behaviour-Based robotics</p>
          <h1>Lawnmower simulation</h1>
          <p className="lede">
            Behavior based program controls a lawnmower robot to cut grass, in a way that it keeps the grass in the area as short as possible, and which never runs out of batteries.
          </p>
        </div>

        <div className="control-panel">
          <div className="button-row">
            <button
              className="solid"
              type="button"
              onClick={
                running
                  ? () => {
                      setRunning(false);
                      setRenderState(createRenderState(simulationRef.current, false));
                    }
                  : startSimulation
              }
            >
              {running ? 'Pause' : 'Start'}
            </button>
            <button className="ghost" type="button" onClick={advanceOneMinute} disabled={running}>
              Step 1 min
            </button>
            <button className="ghost" type="button" onClick={resetSimulation}>
              Reset
            </button>
          </div>

          <label className="slider-block" htmlFor="playback-speed">
            <span>Playback speed</span>
            <div className="slider-row">
              <input
                id="playback-speed"
                type="range"
                min="20"
                max="200"
                step="10"
                value={playbackSpeed}
                onChange={(event) => {
                  setPlaybackSpeed(Number(event.target.value));
                }}
              />
              <strong>{playbackSpeed.toFixed(2)}x</strong>
            </div>
          </label>

          <div className="status-strip">
            <StatusPill
              label={telemetry.depleted ? 'Battery lost' : running ? 'Running' : 'Paused'}
              tone={telemetry.depleted ? 'bad' : running ? 'live' : 'idle'}
            />
            <StatusPill
              label={telemetry.sensorsVisible ? 'Adjacent sensors live' : 'Centre-only sensing'}
              tone={telemetry.sensorsVisible ? 'good' : 'idle'}
            />
            <StatusPill label={`${(telemetry.battery * 100).toFixed(0)}% battery`} tone="idle" />
            <StatusPill label={`${telemetry.grassMean.toFixed(2)} mean grass`} tone="idle" />
          </div>
        </div>
      </section>

      <section className="content-grid">
        <article className="canvas-card">

          <canvas ref={canvasRef} width={1140} height={980} />

          <div className="legend">
            <LegendSwatch color="#c96b2c" label="Charging station" />
            <LegendSwatch color="#111827" label="Wall" />
            <LegendSwatch color="#ead9b5" label="Short grass" />
            <LegendSwatch color="#29603d" label="Tall grass" />
            <LegendSwatch color="#f59e0b" label="Trail" />
            <LegendSwatch color="#f97316" label="Target cell" />
          </div>
        </article>

        <aside className="sidebar">
          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Behaviour Stack</p>
                <h2>Decision order</h2>
              </div>
            </div>

            <ul className="notes">
              {BEHAVIOUR_STACK.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Telemetry</p>
                <h2>Live state</h2>
              </div>
            </div>

            <div className="metric-grid">
              <Metric label="Minute" value={telemetry.minute.toFixed(1)} />
              <Metric label="Grid cell" value={`(${telemetry.cellI}, ${telemetry.cellJ})`} />
              <Metric label="Battery" value={`${(telemetry.battery * 100).toFixed(1)}%`} />
              <Metric label="Rain" value={telemetry.rain.toFixed(2)} />
              <Metric label="Grass mean" value={telemetry.grassMean.toFixed(3)} />
              <Metric label="Grass max" value={telemetry.grassMax.toFixed(3)} />
              <Metric label="Command" value={telemetry.commandLabel} />
              <Metric label="Target" value={`(${telemetry.target.i}, ${telemetry.target.j})`} />
              <Metric label="Behaviour" value={telemetry.behaviourTitle} />
              <Metric
                label="Action progress"
                value={`${Math.round(telemetry.actionProgress * 100)}%`}
              />
            </div>
          </article>

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Current Rule</p>
                <h2>{telemetry.behaviourTitle}</h2>
              </div>
            </div>

            <p className="detail-copy">{telemetry.behaviourDetail}</p>
            <p className="detail-copy subtle">
              The TSX port keeps the C++ thresholds, with one browser-side refinement for
              snake-style row transitions at the map edges. The grass-growth, rain, battery, and
              motion models are still local to the frontend version.
            </p>
          </article>
        </aside>
      </section>

      <style jsx>{`
        .page-shell {
          --card-bg: rgba(13, 25, 18, 0.74);
          --card-border: rgba(231, 213, 181, 0.12);
          min-height: 100vh;
          padding: 32px;
          background:
            radial-gradient(circle at top left, rgba(214, 145, 69, 0.18), transparent 30%),
            radial-gradient(circle at bottom right, rgba(57, 104, 71, 0.18), transparent 28%),
            linear-gradient(180deg, #08110d 0%, #101813 44%, #161e17 100%);
          color: #f7f3e8;
          font-family: var(--font-geist-sans), "Avenir Next", "Segoe UI", sans-serif;
        }

        .hero-card,
        .canvas-card,
        .metric-card {
          border: 1px solid var(--card-border);
          background: var(--card-bg);
          backdrop-filter: blur(18px);
          box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }

        .hero-card {
          display: grid;
          grid-template-columns: minmax(0, 1.65fr) minmax(320px, 0.85fr);
          gap: 24px;
          padding: 28px;
          border-radius: 28px;
          margin-bottom: 24px;
        }

        .hero-copy h1,
        .section-head h2 {
          margin: 0;
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-weight: 600;
          letter-spacing: -0.03em;
        }

        .hero-copy h1 {
          font-size: clamp(2.4rem, 4vw, 4.5rem);
          line-height: 0.96;
          max-width: 11ch;
        }

        .eyebrow,
        .mini-label {
          margin: 0 0 10px;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          font-size: 0.72rem;
          color: rgba(247, 243, 232, 0.7);
        }

        .lede,
        .annotation,
        .detail-copy,
        .notes,
        .history-meta {
          color: rgba(247, 243, 232, 0.82);
        }

        .lede {
          max-width: 64ch;
          line-height: 1.68;
          margin: 18px 0 0;
          font-size: 1rem;
        }

        .detail-copy {
          margin: 0;
          line-height: 1.65;
        }

        .detail-copy.subtle {
          margin-top: 12px;
          color: rgba(247, 243, 232, 0.65);
        }

        .control-panel {
          display: flex;
          flex-direction: column;
          gap: 20px;
          justify-content: space-between;
          padding: 18px;
          border-radius: 24px;
          background: linear-gradient(180deg, rgba(33, 48, 34, 0.82), rgba(15, 24, 18, 0.88));
        }

        .button-row {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }

        button {
          border: 0;
          border-radius: 999px;
          padding: 13px 20px;
          font: inherit;
          cursor: pointer;
          transition: transform 140ms ease, opacity 140ms ease, background 140ms ease;
        }

        button:hover:not(:disabled) {
          transform: translateY(-1px);
        }

        button:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }

        .solid {
          background: linear-gradient(135deg, #d97706, #f59e0b);
          color: #121008;
          font-weight: 700;
        }

        .ghost {
          background: rgba(255, 255, 255, 0.06);
          color: #f7f3e8;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .slider-block {
          display: flex;
          flex-direction: column;
          gap: 10px;
          color: #efe8d3;
        }

        .slider-row {
          display: grid;
          grid-template-columns: 1fr auto;
          gap: 12px;
          align-items: center;
        }

        input[type='range'] {
          width: 100%;
          accent-color: #d97706;
        }

        .status-strip {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .content-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.55fr) minmax(330px, 0.72fr);
          gap: 24px;
          align-items: start;
        }

        .canvas-card,
        .metric-card {
          border-radius: 24px;
        }

        .canvas-card {
          padding: 18px 18px 16px;
        }

        .metric-card {
          padding: 18px;
        }

        .sidebar {
          display: grid;
          gap: 24px;
        }

        .section-head {
          display: flex;
          align-items: end;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 18px;
        }

        .section-head.compact {
          margin-bottom: 14px;
        }

        .annotation {
          margin: 0;
          max-width: 34ch;
          font-size: 0.92rem;
          line-height: 1.45;
          text-align: right;
        }

        canvas {
          width: 100%;
          height: auto;
          display: block;
          border-radius: 22px;
          background:
            radial-gradient(circle at 20% 18%, rgba(255, 255, 255, 0.08), transparent 22%),
            linear-gradient(180deg, #e8efe4 0%, #d7e1cf 100%);
        }

        .legend {
          display: flex;
          gap: 18px;
          flex-wrap: wrap;
          padding: 14px 4px 2px;
        }

        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }

        .sensor-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .sensor-cell {
          display: flex;
          min-height: 76px;
          flex-direction: column;
          justify-content: space-between;
          gap: 6px;
          padding: 12px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .sensor-cell.centre {
          background: rgba(217, 119, 6, 0.12);
          border-color: rgba(245, 158, 11, 0.34);
        }

        .sensor-cell.muted {
          opacity: 0.55;
        }

        .sensor-cell span {
          color: rgba(247, 243, 232, 0.72);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .sensor-cell strong {
          font-size: 1.02rem;
          color: #f7f3e8;
        }

        .history-list,
        .notes {
          margin: 0;
          padding-left: 18px;
          line-height: 1.62;
        }

        .history-list {
          list-style: none;
          padding: 0;
          display: grid;
          gap: 10px;
        }

        .history-list li {
          display: grid;
          gap: 2px;
          padding: 12px 14px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .history-minute {
          color: rgba(247, 243, 232, 0.62);
          font-size: 0.82rem;
        }

        .history-title {
          color: #f7f3e8;
          font-size: 0.95rem;
        }

        .history-meta {
          font-size: 0.82rem;
        }

        .history-empty {
          color: rgba(247, 243, 232, 0.68);
        }

        code {
          font-family: var(--font-geist-mono), "SFMono-Regular", "Menlo", monospace;
          background: rgba(255, 255, 255, 0.08);
          padding: 2px 6px;
          border-radius: 8px;
        }

        @media (max-width: 1180px) {
          .hero-card,
          .content-grid {
            grid-template-columns: 1fr;
          }

          .annotation {
            text-align: left;
            max-width: none;
          }
        }

        @media (max-width: 760px) {
          .page-shell {
            padding: 18px;
          }

          .hero-card,
          .canvas-card,
          .metric-card {
            border-radius: 20px;
          }

          .hero-card {
            padding: 20px;
          }

          .metric-grid,
          .sensor-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </main>
  );
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: 'good' | 'bad' | 'live' | 'idle';
}) {
  return (
    <span className={`status-pill ${tone}`}>
      {label}

      <style jsx>{`
        .status-pill {
          display: inline-flex;
          align-items: center;
          padding: 9px 14px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          font-size: 0.88rem;
          letter-spacing: 0.03em;
        }

        .good {
          background: rgba(34, 197, 94, 0.16);
          color: #d1fae5;
        }

        .bad {
          background: rgba(239, 68, 68, 0.18);
          color: #fee2e2;
        }

        .live {
          background: rgba(217, 119, 6, 0.18);
          color: #ffedd5;
        }

        .idle {
          background: rgba(255, 255, 255, 0.06);
          color: rgba(247, 243, 232, 0.92);
        }
      `}</style>
    </span>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="legend-item">
      <span className="swatch" style={{ backgroundColor: color }} />
      <span>{label}</span>

      <style jsx>{`
        .legend-item {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          color: rgba(247, 243, 232, 0.88);
          font-size: 0.92rem;
        }

        .swatch {
          width: 14px;
          height: 14px;
          border-radius: 999px;
          box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.16);
        }
      `}</style>
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>

      <style jsx>{`
        .metric {
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding: 14px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
          min-height: 88px;
        }

        span {
          color: rgba(247, 243, 232, 0.72);
          font-size: 0.82rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        strong {
          color: #f7f3e8;
          font-size: 1rem;
          line-height: 1.35;
        }
      `}</style>
    </div>
  );
}

function createSimulationState(): SimulationState {
  const initialDecision: BehaviourDecision = {
    command: 0,
    title: 'Charge at dock',
    detail: 'The mower starts at the charging station with a partial battery.',
    target: { ...CHARGING_STATION },
  };
  const grass = createInitialGrassGrid();
  const initialState: SimulationState = {
    grass,
    cell: { ...CHARGING_STATION },
    pose: {
      x: CHARGING_STATION.i + 0.5,
      y: CHARGING_STATION.j + 0.5,
      yaw: 0,
    },
    battery: INITIAL_BATTERY,
    rain: INITIAL_RAIN,
    time: 0,
    previousCommand: 0,
    lastTravelCommand: 4,
    action: null,
    lastSensors: {} as SensorsSnapshot,
    lastDecision: initialDecision,
    status: computeStatus(grass),
    trail: [cellCenter(CHARGING_STATION)],
    recentActions: [],
    rngState: RANDOM_SEED,
    bladePhase: 0,
    depleted: false,
  };

  initialState.lastSensors = buildSensors(initialState);
  initialState.lastDecision = decideNextAction(
    initialState.lastSensors,
    initialState.lastTravelCommand,
  );

  return initialState;
}

function createRenderState(simulation: SimulationState, running: boolean): RenderState {
  return {
    telemetry: createTelemetry(simulation, running),
    activeSensors: { ...(simulation.action?.sensors ?? simulation.lastSensors) },
    recentActions: simulation.recentActions.map((entry) => ({ ...entry })),
  };
}

function createTelemetry(simulation: SimulationState, running: boolean): Telemetry {
  const activeAction = simulation.action;
  const sensors = activeAction?.sensors ?? simulation.lastSensors;
  const decision = activeAction?.decision ?? simulation.lastDecision;
  const command = COMMANDS[decision.command];

  return {
    minute: simulation.time + (activeAction ? activeAction.elapsed / activeAction.duration : 0),
    cellI: simulation.cell.i,
    cellJ: simulation.cell.j,
    battery: simulation.battery,
    rain: simulation.rain,
    grassMean: simulation.status.grassMean,
    grassMax: simulation.status.grassMax,
    commandLabel: simulation.depleted ? 'Battery depleted' : command.label,
    behaviourTitle: simulation.depleted ? 'Battery depleted' : decision.title,
    behaviourDetail: simulation.depleted
      ? 'The mower exhausted its battery before it could recover to the charging station.'
      : decision.detail,
    sensorsVisible: sensors.nearbyVisible,
    target: activeAction?.toCell ?? decision.target,
    actionProgress: running && activeAction ? activeAction.elapsed / activeAction.duration : 0,
    depleted: simulation.depleted,
  };
}

function stepSimulation(simulation: SimulationState, dt: number) {
  if (simulation.depleted) {
    return;
  }

  simulation.bladePhase += dt * 12;

  let remaining = dt;
  while (remaining > 0) {
    if (!simulation.action) {
      simulation.action = prepareAction(simulation);
    }

    const action = simulation.action;
    const step = Math.min(remaining, action.duration - action.elapsed);
    action.elapsed += step;

    const progress = clamp(action.elapsed / action.duration, 0, 1);
    simulation.pose = interpolatePose(action.startPose, action.endPose, easeInOut(progress));

    if (action.elapsed >= action.duration - EPSILON) {
      simulation.pose = { ...action.endPose };
      finalizeAction(simulation, action);
      simulation.action = null;
    }

    remaining -= step;
  }
}

function runSingleMinute(simulation: SimulationState) {
  if (simulation.depleted) {
    return;
  }

  const action = simulation.action ?? prepareAction(simulation);
  simulation.pose = { ...action.endPose };
  finalizeAction(simulation, action);
  simulation.action = null;
}

function prepareAction(simulation: SimulationState): ActionAnimation {
  const sensors = buildSensors(simulation);
  const decision = decideNextAction(sensors, simulation.lastTravelCommand);
  const fromCell = { ...simulation.cell };
  const toCell = resolveTargetCell(fromCell, decision.command);
  const startPose = { ...simulation.pose };
  const intended = COMMANDS[decision.command];
  const nextYaw =
    decision.command === 0 ? simulation.pose.yaw : Math.atan2(intended.dj, intended.di);

  simulation.lastSensors = sensors;
  simulation.lastDecision = decision;

  return {
    decision,
    sensors,
    fromCell,
    toCell,
    startPose,
    endPose: {
      x: toCell.i + 0.5,
      y: toCell.j + 0.5,
      yaw: nextYaw,
    },
    elapsed: 0,
    duration: ACTION_DURATION,
  };
}

function finalizeAction(simulation: SimulationState, action: ActionAnimation) {
  const { decision, fromCell, toCell } = action;

  if (decision.command === 0) {
    if (isChargingStation(fromCell.i, fromCell.j)) {
      simulation.battery = clamp(simulation.battery + CHARGE_RATE, 0, CHARGING_LEVEL);
    } else {
      simulation.battery = clamp(simulation.battery - IDLE_DRAIN, 0, CHARGING_LEVEL);
    }
  } else {
    simulation.battery = clamp(simulation.battery - MOVE_DRAIN, 0, CHARGING_LEVEL);
  }

  if (decision.command === 0) {
    if (simulation.rain < RAIN_LIMIT) {
      setGrass(simulation.grass, fromCell, 0);
    }
  } else if (simulation.rain < RAIN_LIMIT) {
    trimGrass(simulation.grass, fromCell, SOURCE_PASS_CUT_FACTOR);
    trimGrass(simulation.grass, toCell, PASS_CUT_FACTOR);
  }

  simulation.cell = { ...toCell };
  simulation.pose = { ...action.endPose };

  growGrass(simulation.grass, simulation.time, simulation.rain);
  simulation.rain = advanceRain(simulation);
  simulation.time += 1;
  simulation.previousCommand = decision.command;
  if (decision.command !== 0) {
    simulation.lastTravelCommand = decision.command;
  }
  simulation.status = computeStatus(simulation.grass);
  simulation.lastSensors = buildSensors(simulation);
  simulation.lastDecision = decision;
  simulation.trail.push(cellCenter(toCell));
  if (simulation.trail.length > MAX_TRAIL_POINTS) {
    simulation.trail.shift();
  }

  simulation.recentActions.unshift({
    minute: simulation.time,
    command: decision.command,
    title: decision.title,
    battery: simulation.battery,
    rain: simulation.rain,
  });
  simulation.recentActions = simulation.recentActions.slice(0, MAX_HISTORY);

  if (simulation.battery <= EPSILON) {
    simulation.depleted = true;
  }
}

function buildSensors(simulation: Pick<SimulationState, 'battery' | 'cell' | 'grass' | 'previousCommand' | 'rain' | 'time'>): SensorsSnapshot {
  const nearbyVisible = simulation.previousCommand === 0;
  const { i, j } = simulation.cell;
  const neighbourValue = (di: number, dj: number) =>
    nearbyVisible ? readGrass(simulation.grass, i + di, j + dj) : -1;

  return {
    i,
    j,
    time: simulation.time,
    battery: simulation.battery,
    rain: simulation.rain,
    nearbyVisible,
    grassTopLeft: neighbourValue(-1, -1),
    grassTopCentre: neighbourValue(0, -1),
    grassTopRight: neighbourValue(1, -1),
    grassLeft: neighbourValue(-1, 0),
    grassCentre: readGrass(simulation.grass, i, j),
    grassRight: neighbourValue(1, 0),
    grassBottomLeft: neighbourValue(-1, 1),
    grassBottomCentre: neighbourValue(0, 1),
    grassBottomRight: neighbourValue(1, 1),
  };
}

function decideNextAction(
  sensors: SensorsSnapshot,
  lastTravelCommand: CommandId,
): BehaviourDecision {
  const current = { i: sensors.i, j: sensors.j };
  const lowBatteryReserve = Math.max(
    LOW_BATTERY_LEVEL,
    estimateReturnHomeBatteryReserve(current),
  );

  if (sensors.i === 0 && sensors.j === 0 && sensors.battery < CHARGING_LEVEL - EPSILON) {
    return {
      command: 0,
      title: 'Charge at dock',
      detail: 'Priority one is active: the mower stays at the charging station until the battery is full.',
      target: current,
    };
  }

  if (sensors.battery <= lowBatteryReserve) {
    if (sensors.i > 0 && sensors.j > 0) {
      if (sensors.i < WALL_OPENING_COLUMN && sensors.j > WALL_ROW + 1) {
        return {
          command: 3,
          title: 'Low-battery return',
          detail: `Battery dropped below the computed return reserve (${lowBatteryReserve.toFixed(2)}), so the mower climbs diagonally toward the wall opening.`,
          target: resolveTargetCell(current, 3),
        };
      }

      if (sensors.i < WALL_OPENING_COLUMN && sensors.j === WALL_ROW + 1) {
        return {
          command: 4,
          title: 'Low-battery return',
          detail: `Battery is in the return-home reserve band (${lowBatteryReserve.toFixed(2)}), so the mower slides right toward the wall opening.`,
          target: resolveTargetCell(current, 4),
        };
      }

      if (sensors.i === WALL_OPENING_COLUMN && sensors.j > WALL_ROW) {
        return {
          command: 2,
          title: 'Low-battery return',
          detail: `Battery is conserving the remaining path budget (${lowBatteryReserve.toFixed(2)}), so the mower moves upward through the gap.`,
          target: resolveTargetCell(current, 2),
        };
      }

      return {
        command: 1,
        title: 'Low-battery return',
        detail: `The mower is preserving its computed reserve (${lowBatteryReserve.toFixed(2)}) and takes the final diagonal path back toward the dock.`,
        target: resolveTargetCell(current, 1),
      };
    }

    if (sensors.j > 0) {
      return {
        command: 2,
        title: 'Low-battery return',
        detail: `Already on the left border, the mower uses the remaining reserve (${lowBatteryReserve.toFixed(2)}) to reach the station row.`,
        target: resolveTargetCell(current, 2),
      };
    }

    if (sensors.i > 0) {
      return {
        command: 8,
        title: 'Low-battery return',
        detail: `Already on the top row, the mower uses the remaining reserve (${lowBatteryReserve.toFixed(2)}) to slide left toward the charging station.`,
        target: resolveTargetCell(current, 8),
      };
    }
  }

  if (sensors.grassCentre > GRASS_CENTRE_LIMIT && sensors.rain < RAIN_LIMIT) {
    return {
      command: 0,
      title: 'Cut current cell',
      detail: 'The current cell is still tall enough, so the mower stays and cuts it completely.',
      target: current,
    };
  }

  if (
    sensors.nearbyVisible &&
    isRightwardTravel(lastTravelCommand) &&
    current.i === GRID_COLUMNS - 1 &&
    sensors.grassBottomLeft > 0.1
  ) {
    return {
      command: 7,
      title: 'Row transition search',
      detail: 'After a rightward sweep reaches the map edge, the mower drops diagonally into the next row instead of descending the same edge column.',
      target: resolveTargetCell(current, 7),
    };
  }

  if (
    sensors.nearbyVisible &&
    isLeftwardTravel(lastTravelCommand) &&
    current.i === 0 &&
    sensors.grassBottomRight > 0.4
  ) {
    return {
      command: 5,
      title: 'Row transition search',
      detail: 'After a leftward sweep reaches the map edge, the mower drops diagonally into the next row instead of descending the same edge column.',
      target: resolveTargetCell(current, 5),
    };
  }

  for (const candidate of SENSOR_PRIORITY) {
    if (sensors[candidate.key] > candidate.threshold) {
      return {
        command: candidate.command,
        title: 'Follow nearby grass',
        detail: candidate.detail,
        target: resolveTargetCell(current, candidate.command),
      };
    }
  }

  if (sensors.j === WALL_GUIDE_ROW && sensors.i <= WALL_OPENING_COLUMN) {
    return {
      command: 4,
      title: 'Wall-follow search',
      detail: 'No stronger signal is available, so the mower follows the wall to discover new growth.',
      target: resolveTargetCell(current, 4),
    };
  }

  return {
    command: 0,
    title: 'Refresh sensors',
    detail: 'Nearby cells are unknown or not interesting enough, so the mower pauses to request a new adjacent-cell scan.',
    target: current,
  };
}

function isRightwardTravel(command: CommandId) {
  return command === 3 || command === 4 || command === 5;
}

function isLeftwardTravel(command: CommandId) {
  return command === 1 || command === 7 || command === 8;
}

function drawScene(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  simulation: SimulationState,
) {
  const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, '#e4ebdf');
  gradient.addColorStop(1, '#ccd8c2');
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);

  const pad = 78;
  const cellSize = Math.min(
    (canvas.width - pad * 2) / GRID_COLUMNS,
    (canvas.height - pad * 2) / GRID_ROWS,
  );
  const gridWidth = cellSize * GRID_COLUMNS;
  const gridHeight = cellSize * GRID_ROWS;
  const gridLeft = (canvas.width - gridWidth) / 2;
  const gridTop = (canvas.height - gridHeight) / 2;

  const pointToScreen = (point: Point): Point => ({
    x: gridLeft + point.x * cellSize,
    y: gridTop + point.y * cellSize,
  });

  const cellToScreen = (cell: GridPoint): Point => ({
    x: gridLeft + cell.i * cellSize,
    y: gridTop + cell.j * cellSize,
  });

  fillRoundedRect(context, gridLeft - 14, gridTop - 14, gridWidth + 28, gridHeight + 28, 26);
  context.fillStyle = 'rgba(245, 239, 223, 0.78)';
  fillRoundedRect(context, gridLeft - 14, gridTop - 14, gridWidth + 28, gridHeight + 28, 26);

  drawCells(context, simulation, cellSize, cellToScreen);
  drawAxisLabels(context, cellSize, gridLeft, gridTop, gridWidth, gridHeight);

  const activeAction = simulation.action;
  const sensors = activeAction?.sensors ?? simulation.lastSensors;
  const decision = activeAction?.decision ?? simulation.lastDecision;
  const target = activeAction?.toCell ?? decision.target;

  drawTrail(context, simulation.trail, pointToScreen, cellSize);
  drawTargetCell(context, target, cellSize, cellToScreen);
  drawSensorReadings(context, sensors, simulation.cell, cellSize, cellToScreen);
  drawChargingStation(context, cellSize, cellToScreen);
  drawVehicle(context, simulation, pointToScreen, cellSize);
}

function drawCells(
  context: CanvasRenderingContext2D,
  simulation: SimulationState,
  cellSize: number,
  cellToScreen: (cell: GridPoint) => Point,
) {
  for (let j = 0; j < GRID_ROWS; j += 1) {
    for (let i = 0; i < GRID_COLUMNS; i += 1) {
      const topLeft = cellToScreen({ i, j });
      const value = simulation.grass[j][i];

      context.fillStyle = isWallCell(i, j)
        ? '#111827'
        : isChargingStation(i, j)
          ? '#c96b2c'
          : grassColor(value, simulation.rain);
      context.fillRect(topLeft.x, topLeft.y, cellSize, cellSize);

      context.strokeStyle = isWallCell(i, j)
        ? 'rgba(255, 255, 255, 0.08)'
        : 'rgba(11, 24, 16, 0.08)';
      context.lineWidth = 1;
      context.strokeRect(topLeft.x, topLeft.y, cellSize, cellSize);
    }
  }

  if (simulation.rain > 0.04) {
    context.fillStyle = `rgba(59, 130, 246, ${clamp(simulation.rain * 0.14, 0.02, 0.12)})`;
    fillRoundedRect(
      context,
      cellToScreen({ i: 0, j: 0 }).x,
      cellToScreen({ i: 0, j: 0 }).y,
      GRID_COLUMNS * cellSize,
      GRID_ROWS * cellSize,
      18,
    );
  }
}

function drawAxisLabels(
  context: CanvasRenderingContext2D,
  cellSize: number,
  gridLeft: number,
  gridTop: number,
  gridWidth: number,
  gridHeight: number,
) {
  context.fillStyle = 'rgba(22, 33, 24, 0.72)';
  context.font = '500 11px var(--font-geist-mono), monospace';
  context.textAlign = 'center';

  for (let i = 0; i < GRID_COLUMNS; i += 5) {
    context.fillText(`${i}`, gridLeft + i * cellSize + cellSize / 2, gridTop - 14);
  }

  context.textAlign = 'right';
  for (let j = 0; j < GRID_ROWS; j += 5) {
    context.fillText(`${j}`, gridLeft - 12, gridTop + j * cellSize + cellSize * 0.65);
  }

  context.strokeStyle = 'rgba(17, 24, 39, 0.1)';
  context.lineWidth = 2;
  context.strokeRect(gridLeft, gridTop, gridWidth, gridHeight);
}

function drawTrail(
  context: CanvasRenderingContext2D,
  trail: Point[],
  pointToScreen: (point: Point) => Point,
  cellSize: number,
) {
  if (trail.length < 2) {
    return;
  }

  context.strokeStyle = 'rgba(245, 158, 11, 0.78)';
  context.lineWidth = Math.max(cellSize * 0.16, 2);
  context.setLineDash([9, 7]);
  context.beginPath();

  const first = pointToScreen(trail[0]);
  context.moveTo(first.x, first.y);

  for (let index = 1; index < trail.length; index += 1) {
    const point = pointToScreen(trail[index]);
    context.lineTo(point.x, point.y);
  }

  context.stroke();
  context.setLineDash([]);
}

function drawTargetCell(
  context: CanvasRenderingContext2D,
  target: GridPoint,
  cellSize: number,
  cellToScreen: (cell: GridPoint) => Point,
) {
  const topLeft = cellToScreen(target);
  context.strokeStyle = '#f97316';
  context.lineWidth = 2.5;
  context.strokeRect(topLeft.x + 2, topLeft.y + 2, cellSize - 4, cellSize - 4);

  context.fillStyle = 'rgba(249, 115, 22, 0.18)';
  context.fillRect(topLeft.x + 3, topLeft.y + 3, cellSize - 6, cellSize - 6);
}

function drawSensorReadings(
  context: CanvasRenderingContext2D,
  sensors: SensorsSnapshot,
  cell: GridPoint,
  cellSize: number,
  cellToScreen: (cell: GridPoint) => Point,
) {
  const sensorFont = Math.max(cellSize * 0.25, 8);

  for (const sensor of SENSOR_LAYOUT) {
    const targetI = cell.i + sensor.di;
    const targetJ = cell.j + sensor.dj;
    if (!isInsideGrid(targetI, targetJ)) {
      continue;
    }

    const value = sensors[sensor.key];
    const topLeft = cellToScreen({ i: targetI, j: targetJ });

    if (sensor.key === 'grassCentre') {
      context.strokeStyle = 'rgba(12, 74, 110, 0.85)';
      context.lineWidth = 2;
      context.strokeRect(topLeft.x + 1.5, topLeft.y + 1.5, cellSize - 3, cellSize - 3);
    }

    if (sensor.key !== 'grassCentre' && !sensors.nearbyVisible) {
      continue;
    }

    context.fillStyle =
      sensor.key === 'grassCentre'
        ? 'rgba(12, 74, 110, 0.82)'
        : 'rgba(15, 23, 42, 0.72)';
    fillRoundedRect(
      context,
      topLeft.x + cellSize * 0.08,
      topLeft.y + cellSize * 0.1,
      cellSize * 0.84,
      cellSize * 0.34,
      cellSize * 0.12,
    );

    context.fillStyle = '#f8fafc';
    context.font = `600 ${sensorFont}px var(--font-geist-mono), monospace`;
    context.textAlign = 'center';
    context.fillText(
      value < 0 ? 'n/a' : value.toFixed(2),
      topLeft.x + cellSize * 0.5,
      topLeft.y + cellSize * 0.34,
    );
  }
}

function drawChargingStation(
  context: CanvasRenderingContext2D,
  cellSize: number,
  cellToScreen: (cell: GridPoint) => Point,
) {
  const topLeft = cellToScreen(CHARGING_STATION);
  context.fillStyle = 'rgba(255, 244, 213, 0.28)';
  fillRoundedRect(context, topLeft.x + 3, topLeft.y + 3, cellSize - 6, cellSize - 6, 6);

  context.strokeStyle = '#fff1c9';
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(topLeft.x + cellSize * 0.35, topLeft.y + cellSize * 0.25);
  context.lineTo(topLeft.x + cellSize * 0.35, topLeft.y + cellSize * 0.75);
  context.lineTo(topLeft.x + cellSize * 0.68, topLeft.y + cellSize * 0.75);
  context.lineTo(topLeft.x + cellSize * 0.68, topLeft.y + cellSize * 0.25);
  context.stroke();
}

function drawVehicle(
  context: CanvasRenderingContext2D,
  simulation: SimulationState,
  pointToScreen: (point: Point) => Point,
  cellSize: number,
) {
  const center = pointToScreen({ x: simulation.pose.x, y: simulation.pose.y });
  const bodyLength = cellSize * 0.88;
  const bodyWidth = cellSize * 0.58;

  context.save();
  context.translate(center.x, center.y);
  context.rotate(simulation.pose.yaw);

  context.fillStyle = simulation.depleted ? '#ef4444' : '#f7b267';
  context.beginPath();
  context.roundRect(-bodyLength / 2, -bodyWidth / 2, bodyLength, bodyWidth, 8);
  context.fill();

  context.fillStyle = '#1f2937';
  context.beginPath();
  context.moveTo(bodyLength / 2, 0);
  context.lineTo(bodyLength / 2 - 12, -8);
  context.lineTo(bodyLength / 2 - 12, 8);
  context.closePath();
  context.fill();

  context.fillStyle = simulation.action?.decision.command === 0 ? '#166534' : '#0f172a';
  context.beginPath();
  context.arc(0, 0, bodyWidth * 0.28, 0, Math.PI * 2);
  context.fill();

  if (simulation.action?.decision.command === 0) {
    context.strokeStyle = 'rgba(254, 240, 138, 0.92)';
    context.lineWidth = 2;
    for (let blade = 0; blade < 3; blade += 1) {
      const angle = simulation.bladePhase + (blade * Math.PI * 2) / 3;
      context.beginPath();
      context.moveTo(0, 0);
      context.lineTo(Math.cos(angle) * bodyWidth * 0.34, Math.sin(angle) * bodyWidth * 0.34);
      context.stroke();
    }
  }

  context.restore();
}

function drawHud(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  simulation: SimulationState,
  decision: BehaviourDecision,
  sensors: SensorsSnapshot,
) {
  context.fillStyle = 'rgba(12, 21, 16, 0.88)';
  fillRoundedRect(context, 20, 20, 342, 126, 20);

  context.fillStyle = '#f7f3e8';
  context.font = '600 20px var(--font-geist-sans), sans-serif';
  context.fillText(simulation.depleted ? 'Battery depleted' : decision.title, 36, 52);

  context.fillStyle = 'rgba(247, 243, 232, 0.8)';
  context.font = '500 14px var(--font-geist-sans), sans-serif';
  context.fillText(`command      ${COMMANDS[decision.command].label}`, 36, 82);
  context.fillText(`cell         (${simulation.cell.i}, ${simulation.cell.j})`, 36, 104);
  context.fillText(
    `sensors      ${sensors.nearbyVisible ? 'adjacent live' : 'centre only'}`,
    36,
    126,
  );

  context.fillStyle = 'rgba(12, 21, 16, 0.88)';
  fillRoundedRect(context, canvas.width - 224, 20, 204, 140, 20);

  context.fillStyle = '#f7f3e8';
  context.font = '600 16px var(--font-geist-sans), sans-serif';
  context.fillText(`t = ${simulation.time} min`, canvas.width - 206, 50);

  context.fillStyle = 'rgba(247, 243, 232, 0.8)';
  context.font = '500 14px var(--font-geist-sans), sans-serif';
  context.fillText(
    `battery     ${(simulation.battery * 100).toFixed(0)}%`,
    canvas.width - 206,
    80,
  );
  context.fillText(`rain        ${simulation.rain.toFixed(2)}`, canvas.width - 206, 104);
  context.fillText(
    `mean grass  ${simulation.status.grassMean.toFixed(2)}`,
    canvas.width - 206,
    128,
  );
  context.fillText(
    `max grass   ${simulation.status.grassMax.toFixed(2)}`,
    canvas.width - 206,
    152,
  );
}

function createInitialGrassGrid() {
  return Array.from({ length: GRID_ROWS }, (_, j) =>
    Array.from({ length: GRID_COLUMNS }, (_, i) =>
      isWallCell(i, j) || isChargingStation(i, j) ? 0 : 1,
    ),
  );
}

function computeStatus(grass: number[][]): StatusSnapshot {
  let grassSum = 0;
  let grassMax = 0;
  let cells = 0;

  for (let j = 0; j < GRID_ROWS; j += 1) {
    for (let i = 0; i < GRID_COLUMNS; i += 1) {
      if (isWallCell(i, j) || isChargingStation(i, j)) {
        continue;
      }

      grassSum += grass[j][i];
      grassMax = Math.max(grassMax, grass[j][i]);
      cells += 1;
    }
  }

  return {
    grassMean: cells > 0 ? grassSum / cells : 0,
    grassMax,
  };
}

function cellCenter(cell: GridPoint): Point {
  return {
    x: cell.i + 0.5,
    y: cell.j + 0.5,
  };
}

function resolveTargetCell(cell: GridPoint, command: CommandId): GridPoint {
  const delta = COMMANDS[command];
  const nextI = cell.i + delta.di;
  const nextJ = cell.j + delta.dj;

  if (command === 0 || !isInsideGrid(nextI, nextJ) || isWallCell(nextI, nextJ)) {
    return { ...cell };
  }

  return {
    i: nextI,
    j: nextJ,
  };
}

function estimateReturnHomeBatteryReserve(cell: GridPoint) {
  return estimateReturnHomeMoves(cell) * MOVE_DRAIN + RETURN_SAFETY_MARGIN;
}

function estimateReturnHomeMoves(start: GridPoint) {
  let steps = 0;
  let current = { ...start };

  while (!isChargingStation(current.i, current.j) && steps < GRID_COLUMNS + GRID_ROWS + 20) {
    const command = chooseReturnHomeCommand(current);
    const next = resolveTargetCell(current, command);

    if (next.i === current.i && next.j === current.j) {
      break;
    }

    current = next;
    steps += 1;
  }

  return steps;
}

function chooseReturnHomeCommand(cell: GridPoint): CommandId {
  if (cell.i > 0 && cell.j > 0) {
    if (cell.i < WALL_OPENING_COLUMN && cell.j > WALL_ROW + 1) {
      return 3;
    }

    if (cell.i < WALL_OPENING_COLUMN && cell.j === WALL_ROW + 1) {
      return 4;
    }

    if (cell.i === WALL_OPENING_COLUMN && cell.j > WALL_ROW) {
      return 2;
    }

    return 1;
  }

  if (cell.j > 0) {
    return 2;
  }

  if (cell.i > 0) {
    return 8;
  }

  return 0;
}

function isInsideGrid(i: number, j: number) {
  return i >= 0 && j >= 0 && i < GRID_COLUMNS && j < GRID_ROWS;
}

function isWallCell(i: number, j: number) {
  return j === WALL_ROW && i >= 0 && i <= WALL_END_COLUMN;
}

function isChargingStation(i: number, j: number) {
  return i === CHARGING_STATION.i && j === CHARGING_STATION.j;
}

function readGrass(grass: number[][], i: number, j: number): number {
  if (!isInsideGrid(i, j) || isWallCell(i, j) || isChargingStation(i, j)) {
    return 0;
  }

  return grass[j][i];
}

function setGrass(grass: number[][], cell: GridPoint, value: number) {
  if (isChargingStation(cell.i, cell.j) || isWallCell(cell.i, cell.j)) {
    return;
  }

  grass[cell.j][cell.i] = clamp(value, 0, 1);
}

function trimGrass(grass: number[][], cell: GridPoint, factor: number) {
  if (isChargingStation(cell.i, cell.j) || isWallCell(cell.i, cell.j)) {
    return;
  }

  grass[cell.j][cell.i] = clamp(grass[cell.j][cell.i] * factor, 0, 1);
}

function growGrass(grass: number[][], time: number, rain: number) {
  for (let j = 0; j < GRID_ROWS; j += 1) {
    for (let i = 0; i < GRID_COLUMNS; i += 1) {
      if (isWallCell(i, j) || isChargingStation(i, j)) {
        grass[j][i] = 0;
        continue;
      }

      const terrainBias = 0;
      grass[j][i] = clamp(grass[j][i] + BASE_GROWTH + rain * RAIN_GROWTH_FACTOR + terrainBias, 0, 1);
    }
  }
}

function advanceRain(simulation: Pick<SimulationState, 'rain' | 'rngState'>): number {
  const firstRoll = nextRandom(simulation);
  const secondRoll = nextRandom(simulation);
  const dryThreshold = 0.06 * RAIN_AMOUNT_SCALE;

  let nextRain = simulation.rain;
  if (nextRain < dryThreshold) {
    if (firstRoll < 0.01) {
      nextRain = (0.06 + secondRoll * 0.14) * RAIN_AMOUNT_SCALE;
    } else {
      nextRain = Math.max(0, nextRain - 0.065 * RAIN_AMOUNT_SCALE);
    }
  } else {
    const drift = (firstRoll - 0.68) * (0.055 * RAIN_AMOUNT_SCALE);
    nextRain = clamp(nextRain + drift, 0, 1);
    if (secondRoll < 0.46) {
      nextRain *= 0.42;
    }
  }

  return clamp(nextRain, 0, 1);
}

function nextRandom(simulation: Pick<SimulationState, 'rngState'>): number {
  simulation.rngState = (Math.imul(simulation.rngState, 1664525) + 1013904223) >>> 0;
  return simulation.rngState / 4294967296;
}

function interpolatePose(start: VehiclePose, end: VehiclePose, t: number): VehiclePose {
  return {
    x: start.x + (end.x - start.x) * t,
    y: start.y + (end.y - start.y) * t,
    yaw: lerpAngle(start.yaw, end.yaw, t),
  };
}

function easeInOut(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerpAngle(start: number, end: number, t: number) {
  const delta = normalizeAngle(end - start);
  return normalizeAngle(start + delta * t);
}

function normalizeAngle(angle: number) {
  let normalized = angle;
  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }
  while (normalized < -Math.PI) {
    normalized += Math.PI * 2;
  }
  return normalized;
}

function grassColor(value: number, rain: number) {
  const low: [number, number, number] = [234, 217, 181];
  const high: [number, number, number] = [41, 96, 61];
  const wetTint: [number, number, number] = [64, 116, 167];
  const mixed = mixColor(low, high, Math.pow(clamp(value, 0, 1), 0.88));
  return mixColorTuple(mixed, wetTint, clamp(rain * 0.12, 0, 0.16));
}

function mixColor(a: [number, number, number], b: [number, number, number], t: number) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ] as [number, number, number];
}

function mixColorTuple(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
) {
  const mixed = mixColor(a, b, t);
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
}

function fillRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const boundedRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + boundedRadius, y);
  context.lineTo(x + width - boundedRadius, y);
  context.arcTo(x + width, y, x + width, y + boundedRadius, boundedRadius);
  context.lineTo(x + width, y + height - boundedRadius);
  context.arcTo(x + width, y + height, x + width - boundedRadius, y + height, boundedRadius);
  context.lineTo(x + boundedRadius, y + height);
  context.arcTo(x, y + height, x, y + height - boundedRadius, boundedRadius);
  context.lineTo(x, y + boundedRadius);
  context.arcTo(x, y, x + boundedRadius, y, boundedRadius);
  context.closePath();
  context.fill();
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function formatSensorValue(value: number) {
  return value < 0 ? 'n/a' : value.toFixed(2);
}
