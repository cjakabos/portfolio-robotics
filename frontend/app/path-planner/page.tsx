'use client';

import { useEffect, useRef, useState, type MouseEvent } from 'react';

type Point = {
  x: number;
  y: number;
};

type Segment = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

type GridPoint = {
  i: number;
  j: number;
};

type SensorId = 'frontLeft' | 'front' | 'frontRight' | 'left' | 'right';

type Bounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
};

type Scenario = {
  bounds: Bounds;
  walls: Segment[];
  wallCells: boolean[][];
  blockedCells: boolean[][];
  gridSize: number;
  cellCountX: number;
  cellCountY: number;
  start: Point;
  goal: Point;
  startCell: GridPoint;
  goalCell: GridPoint;
  pathCells: GridPoint[];
  path: Point[];
  pathArcLengths: number[];
  pathLength: number;
  planningClearance: number;
};

type VehicleState = {
  x: number;
  y: number;
  yaw: number;
  speed: number;
  steering: number;
  elapsed: number;
  reachedGoal: boolean;
  collided: boolean;
};

type ControllerSnapshot = {
  nearestIndex: number;
  targetIndex: number;
  targetPoint: Point;
  headingError: number;
  desiredSpeed: number;
  distanceToGoal: number;
  sensorReadings: SensorReading[];
  avoidanceSteering: number;
  minClearance: number;
};

type SimulationState = {
  vehicle: VehicleState;
  controller: ControllerSnapshot;
  trail: Point[];
  trailCarry: number;
};

type Telemetry = {
  x: number;
  y: number;
  yawDeg: number;
  speed: number;
  steeringDeg: number;
  elapsed: number;
  targetIndex: number;
  nearestIndex: number;
  goalDistance: number;
  running: boolean;
  reachedGoal: boolean;
  collided: boolean;
  frontDistance: number;
  leftDistance: number;
  rightDistance: number;
  minClearance: number;
};

type SensorReading = {
  id: SensorId;
  label: string;
  origin: Point;
  angle: number;
  distance: number;
  maxRange: number;
  hitPoint: Point;
  proximity: number;
};

type SensorConfig = {
  id: SensorId;
  label: string;
  offsetX: number;
  offsetY: number;
  relativeAngle: number;
  maxRange: number;
  influenceDistance: number;
};

type SelectionMode = 'start' | 'goal';

const MAP_SOURCE = `
0.0,0.0,0.0,6.0;
0.0,0.0,6.0,0.0;
6.0,0.0,6.0,6.0;
0.0,6.0,6.0,6.0;
1.0,2.0,1.0,5.0;
2.0,0.0,2.0,2.0;
3.0,0.0,3.0,2.0;
3.0,3.0,3.0,6.0;
4.0,0.0,4.0,2.0;
4.0,3.0,4.0,5.0;
5.0,0.0,5.0,1.0;
5.0,4.0,5.0,5.0;
0.0,1.0,1.0,1.0;
1.0,2.0,2.0,2.0;
4.0,2.0,5.0,2.0;
2.0,3.0,3.0,3.0;
4.0,3.0,6.0,3.0;
1.0,5.0,2.0,5.0;
4.0,5.0,5.0,5.0;
`.trim();

const GRID_SIZE = 0.2;
const DEFAULT_START_POINT: Point = { x: 0.3, y: 0.6 };
const DEFAULT_GOAL_POINT: Point = { x: 5.5, y: 0.5 };
const CANVAS_FRAME_PADDING = 24;

const VEHICLE_LENGTH = 0.36;
const VEHICLE_WIDTH = 0.18;
const WHEELBASE = 0.28;
const VEHICLE_RADIUS = Math.hypot(VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2);
const MAX_STEERING = 0.72;
const BASE_SPEED = 0.9;
const MAX_ACCELERATION = 1.4;
const MAX_DECELERATION = 1.8;
const STEERING_RESPONSE = 7.5;
const LOOKAHEAD_DISTANCE = 0.55;
const TRAIL_SPACING = 0.06;
const EPSILON = 1e-9;
const PLANNER_CLEARANCE = VEHICLE_RADIUS + 0.08;

const SENSOR_CONFIGS: SensorConfig[] = [
  {
    id: 'frontLeft',
    label: 'Front-left',
    offsetX: VEHICLE_LENGTH * 0.42,
    offsetY: VEHICLE_WIDTH * 0.32,
    relativeAngle: Math.PI / 4.2,
    maxRange: 1.05,
    influenceDistance: 0.58,
  },
  {
    id: 'front',
    label: 'Front',
    offsetX: VEHICLE_LENGTH * 0.5,
    offsetY: 0,
    relativeAngle: 0,
    maxRange: 1.15,
    influenceDistance: 0.68,
  },
  {
    id: 'frontRight',
    label: 'Front-right',
    offsetX: VEHICLE_LENGTH * 0.42,
    offsetY: -VEHICLE_WIDTH * 0.32,
    relativeAngle: -Math.PI / 4.2,
    maxRange: 1.05,
    influenceDistance: 0.58,
  },
  {
    id: 'left',
    label: 'Left',
    offsetX: 0,
    offsetY: VEHICLE_WIDTH * 0.5,
    relativeAngle: Math.PI / 2,
    maxRange: 0.8,
    influenceDistance: 0.42,
  },
  {
    id: 'right',
    label: 'Right',
    offsetX: 0,
    offsetY: -VEHICLE_WIDTH * 0.5,
    relativeAngle: -Math.PI / 2,
    maxRange: 0.8,
    influenceDistance: 0.42,
  },
];

const NEIGHBORS = [
  { di: 1, dj: 0, cost: 1 },
  { di: -1, dj: 0, cost: 1 },
  { di: 0, dj: 1, cost: 1 },
  { di: 0, dj: -1, cost: 1 },
  { di: 1, dj: 1, cost: Math.SQRT2 },
  { di: 1, dj: -1, cost: Math.SQRT2 },
  { di: -1, dj: 1, cost: Math.SQRT2 },
  { di: -1, dj: -1, cost: Math.SQRT2 },
];

const INITIAL_SCENARIO = buildScenario(DEFAULT_START_POINT, DEFAULT_GOAL_POINT);

export default function Page() {
  const initialSimulation = createSimulationState(INITIAL_SCENARIO);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const scenarioRef = useRef<Scenario>(INITIAL_SCENARIO);
  const simulationRef = useRef<SimulationState>(initialSimulation);
  const pendingStartPointRef = useRef<Point | null>(null);

  const [scenario, setScenario] = useState<Scenario>(() => INITIAL_SCENARIO);
  const [running, setRunning] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(3);
  const [selectionMode, setSelectionMode] = useState<SelectionMode | null>(null);
  const [pendingStartPoint, setPendingStartPoint] = useState<Point | null>(null);
  const [routeMessage, setRouteMessage] = useState(
    'Use Set start and goal, then click once for the start anchor and once for the goal anchor.',
  );
  const [routeMessageTone, setRouteMessageTone] = useState<'good' | 'bad' | 'live' | 'idle'>(
    'idle',
  );
  const [telemetry, setTelemetry] = useState<Telemetry>(() => createTelemetry(initialSimulation, true));

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
      const activeScenario = scenarioRef.current;
      const rawDt = Math.min((timestamp - lastTimestamp) / 1000, 0.05);
      lastTimestamp = timestamp;

      if (running) {
        const scaledDt = rawDt * playbackSpeed;
        stepSimulation(simulationRef.current, activeScenario, scaledDt);
        uiCarry += scaledDt;

        const current = simulationRef.current;
        if (current.vehicle.reachedGoal || current.vehicle.collided) {
          setRunning(false);
          setTelemetry(createTelemetry(current, false));
        } else if (uiCarry >= 0.08) {
          uiCarry = 0;
          setTelemetry(createTelemetry(current, true));
        }
      }

      drawScene(
        context,
        canvas,
        activeScenario,
        simulationRef.current,
        pendingStartPointRef.current,
      );

      frameId = window.requestAnimationFrame(loop);
    };

    drawScene(
      context,
      canvas,
      scenarioRef.current,
      simulationRef.current,
      pendingStartPointRef.current,
    );
    frameId = window.requestAnimationFrame(loop);

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [running, playbackSpeed]);

  const resetSimulation = () => {
    simulationRef.current = createSimulationState(scenarioRef.current);
    setTelemetry(createTelemetry(simulationRef.current, false));
    setRunning(false);
  };

  const startSimulation = () => {
    if (simulationRef.current.vehicle.reachedGoal || simulationRef.current.vehicle.collided) {
      simulationRef.current = createSimulationState(scenarioRef.current);
    }
    setTelemetry(createTelemetry(simulationRef.current, true));
    setRunning(true);
  };

  const applyScenario = (
    nextScenario: Scenario,
    message: string,
    tone: 'good' | 'bad' | 'live' | 'idle',
  ) => {
    scenarioRef.current = nextScenario;
    setScenario(nextScenario);
    simulationRef.current = createSimulationState(nextScenario);
    setTelemetry(createTelemetry(simulationRef.current, false));
    setRunning(false);
    setSelectionMode(null);
    pendingStartPointRef.current = null;
    setPendingStartPoint(null);
    setRouteMessage(message);
    setRouteMessageTone(tone);
  };

  const toggleSelectionMode = () => {
    if (selectionMode !== null) {
      setSelectionMode(null);
      pendingStartPointRef.current = null;
      setPendingStartPoint(null);
      setRouteMessage('Anchor selection cleared. Click Set start and goal to choose a new pair.');
      setRouteMessageTone('idle');
      return;
    }

    setSelectionMode('start');
    pendingStartPointRef.current = null;
    setPendingStartPoint(null);
    setRouteMessage('Click on the map to place a new start point, then click again to place a new goal point.');
    setRouteMessageTone('live');
    setRunning(false);
  };

  const assertAnchorPointIsValid = (point: Point, label: 'Start' | 'Goal') => {
    const activeScenario = scenarioRef.current;
    const cell = locateCell(
      point,
      activeScenario.bounds,
      activeScenario.gridSize,
      activeScenario.cellCountX,
      activeScenario.cellCountY,
    );

    if (activeScenario.wallCells[cell.j][cell.i]) {
      throw new Error(`${label} point lies on a wall.`);
    }

    if (distancePointToWalls(point, activeScenario.walls) < activeScenario.planningClearance) {
      throw new Error(`${label} point violates the planner clearance margin.`);
    }
  };

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!selectionMode) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const pixelPoint = getCanvasPixelFromMouseEvent(event, canvas);
    const clickedPoint = canvasToWorldPoint(pixelPoint, canvas, scenarioRef.current);

    if (!clickedPoint) {
      setRouteMessage(
        `Click inside the map frame to place the ${selectionMode === 'start' ? 'start' : 'goal'} point.`,
      );
      setRouteMessageTone('bad');
      return;
    }

    try {
      if (selectionMode === 'start') {
        assertAnchorPointIsValid(clickedPoint, 'Start');
        pendingStartPointRef.current = clickedPoint;
        setPendingStartPoint(clickedPoint);
        setSelectionMode('goal');
        setRouteMessage(
          `Start set to (${clickedPoint.x.toFixed(2)}, ${clickedPoint.y.toFixed(2)}). Click the map again to place the new goal point.`,
        );
        setRouteMessageTone('live');
        return;
      }

      const nextStart = pendingStartPoint ?? scenarioRef.current.start;
      assertAnchorPointIsValid(clickedPoint, 'Goal');

      const nextScenario = buildScenario(nextStart, clickedPoint);
      applyScenario(
        nextScenario,
        `Updated route from (${nextStart.x.toFixed(2)}, ${nextStart.y.toFixed(2)}) to (${clickedPoint.x.toFixed(2)}, ${clickedPoint.y.toFixed(2)}). Path recalculated and vehicle reset.`,
        'good',
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : `Unable to place the ${selectionMode === 'start' ? 'start' : 'goal'} point there.`;
      setRouteMessage(message);
      setRouteMessageTone('bad');
    }
  };

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Dijkstra&apos;s algorithm / Path planning</p>
          <h1>Path Planner</h1>
          <p className="lede">
            This page ports the core assignment logic from the C++ path follower into a single
            client-side TSX file. The planner still uses an 8-neighbor Dijkstra grid, but the
            browser replaces the OpenDLV motion stack with its own kinematic vehicle model.
          </p>
        </div>

        <div className="control-panel">
          <div className="button-row">
            <button
              className="solid"
              onClick={
                running
                  ? () => {
                      setRunning(false);
                      setTelemetry(createTelemetry(simulationRef.current, false));
                    }
                  : startSimulation
              }
            >
              {running ? 'Pause' : 'Start'}
            </button>
            <button className="ghost" onClick={resetSimulation}>
              Reset
            </button>
          </div>

          <button
            className={`ghost pick-button pick-button-wide ${selectionMode ? 'active' : ''}`}
            onClick={toggleSelectionMode}
            type="button"
          >
            {selectionMode === null
              ? 'Set start and goal'
              : selectionMode === 'start'
                ? 'Picking start...'
                : 'Picking goal...'}
          </button>

          <label className="slider-block" htmlFor="playback-speed">
            <span>Playback speed</span>
            <div className="slider-row">
              <input
                id="playback-speed"
                type="range"
                min="3"
                max="10"
                step="1"
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
              label={telemetry.reachedGoal ? 'Goal reached' : telemetry.collided ? 'Collision' : running ? 'Running' : 'Paused'}
              tone={telemetry.reachedGoal ? 'good' : telemetry.collided ? 'bad' : running ? 'live' : 'idle'}
            />
            <StatusPill
              label={
                selectionMode === 'start'
                  ? 'Picking start'
                  : selectionMode === 'goal'
                    ? 'Picking goal'
                    : 'Route locked'
              }
              tone={selectionMode ? 'live' : 'idle'}
            />
            <StatusPill label={`${scenario.path.length} waypoints`} tone="idle" />
            <StatusPill label={`${scenario.pathLength.toFixed(2)} m path`} tone="idle" />
          </div>

          <p className={`route-note ${routeMessageTone}`}>{routeMessage}</p>
        </div>
      </section>

      <section className="content-grid">
        <article className="canvas-card">
          <div className="section-head">
            <div>
              <p className="mini-label">Simulation</p>
              <h2>Planner + vehicle model</h2>
            </div>
            <p className="annotation">
              Click <code>Set start and goal</code>, then place the start marker followed by the
              goal marker to recompute the route.
            </p>
          </div>

          <canvas
            ref={canvasRef}
            width={1000}
            height={1000}
            onClick={handleCanvasClick}
            className={selectionMode ? 'interactive-canvas' : undefined}
          />

          <div className="legend">
            <LegendSwatch color="#1d4ed8" label="Start" />
            <LegendSwatch color="#f97316" label="Goal" />
            <LegendSwatch color="#f59e0b" label="Planner clearance" />
            <LegendSwatch color="#34d399" label="Dijkstra path" />
            <LegendSwatch color="#f4d35e" label="Vehicle trail" />
            <LegendSwatch color="#f43f5e" label="Lookahead point" />
          </div>
        </article>

        <aside className="sidebar">
          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Telemetry</p>
                <h2>Live state</h2>
              </div>
            </div>

            <div className="metric-grid">
              <Metric label="Position X" value={`${telemetry.x.toFixed(2)} m`} />
              <Metric label="Position Y" value={`${telemetry.y.toFixed(2)} m`} />
              <Metric label="Yaw" value={`${telemetry.yawDeg.toFixed(1)} deg`} />
              <Metric label="Speed" value={`${telemetry.speed.toFixed(2)} m/s`} />
              <Metric label="Steering" value={`${telemetry.steeringDeg.toFixed(1)} deg`} />
              <Metric label="Goal distance" value={`${telemetry.goalDistance.toFixed(2)} m`} />
              <Metric label="Front sensor" value={`${telemetry.frontDistance.toFixed(2)} m`} />
              <Metric label="Left sensor" value={`${telemetry.leftDistance.toFixed(2)} m`} />
              <Metric label="Right sensor" value={`${telemetry.rightDistance.toFixed(2)} m`} />
              <Metric label="Min clearance" value={`${telemetry.minClearance.toFixed(2)} m`} />
              <Metric label="Nearest node" value={`${telemetry.nearestIndex}`} />
              <Metric label="Aim node" value={`${telemetry.targetIndex}`} />
              <Metric label="Elapsed" value={`${telemetry.elapsed.toFixed(1)} s`} />
            </div>
          </article>

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Route Anchors</p>
                <h2>Current plan inputs</h2>
              </div>
            </div>

            <div className="metric-grid">
              <Metric label="Start point" value={`(${scenario.start.x.toFixed(2)}, ${scenario.start.y.toFixed(2)})`} />
              <Metric label="Goal point" value={`(${scenario.goal.x.toFixed(2)}, ${scenario.goal.y.toFixed(2)})`} />
              <Metric label="Start cell" value={`(${scenario.startCell.i}, ${scenario.startCell.j})`} />
              <Metric label="Goal cell" value={`(${scenario.goalCell.i}, ${scenario.goalCell.j})`} />
            </div>
          </article>

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Implementation</p>
                <h2>What changed from the C++ service</h2>
              </div>
            </div>

            <ul className="notes">
              <li>The planner keeps the original grid-based Dijkstra approach with diagonal costs, but now inflates walls by the vehicle footprint.</li>
              <li>A single anchor-selection workflow lets you place a new start point and a new goal point directly on the map.</li>
              <li>The browser computes steering locally from a nearest-point plus lookahead target and virtual sensors.</li>
              <li>The vehicle motion is integrated with a simple bicycle model instead of OpenDLV motor containers.</li>
            </ul>
          </article>
        </aside>
      </section>

      <style jsx>{`
        :global(body) {
          margin: 0;
          background:
            radial-gradient(circle at top left, rgba(71, 85, 105, 0.22), transparent 28%),
            linear-gradient(160deg, #07111d 0%, #0e1726 48%, #1b2638 100%);
          color: #f8fafc;
          font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
        }

        * {
          box-sizing: border-box;
        }

        .page-shell {
          min-height: 100vh;
          padding: 32px;
        }

        .hero-card,
        .canvas-card,
        .metric-card {
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(7, 17, 29, 0.72);
          backdrop-filter: blur(18px);
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        }

        .hero-card {
          display: grid;
          grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
          gap: 24px;
          padding: 28px;
          border-radius: 28px;
          margin-bottom: 24px;
        }

        .hero-copy h1,
        .section-head h2 {
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          font-weight: 600;
          letter-spacing: -0.03em;
        }

        .hero-copy h1 {
          font-size: clamp(2.3rem, 4vw, 4.3rem);
          line-height: 0.94;
          max-width: 12ch;
        }

        .lede,
        .annotation,
        .notes,
        .mini-label,
        .eyebrow {
          color: rgba(226, 232, 240, 0.82);
        }

        .eyebrow,
        .mini-label {
          margin: 0 0 10px;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          font-size: 0.72rem;
        }

        .lede {
          max-width: 64ch;
          line-height: 1.62;
          margin: 18px 0 0;
          font-size: 1rem;
        }

        .control-panel {
          display: flex;
          flex-direction: column;
          gap: 20px;
          justify-content: space-between;
          padding: 18px;
          border-radius: 22px;
          background: linear-gradient(180deg, rgba(20, 34, 55, 0.82), rgba(10, 18, 30, 0.82));
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

        button:hover {
          transform: translateY(-1px);
        }

        .solid {
          background: linear-gradient(135deg, #f97316, #f59e0b);
          color: #081018;
          font-weight: 700;
        }

        .ghost {
          background: rgba(255, 255, 255, 0.06);
          color: #f8fafc;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .pick-button.active {
          background: rgba(59, 130, 246, 0.18);
          border-color: rgba(96, 165, 250, 0.4);
          color: #dbeafe;
        }

        .pick-button-wide {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 100%;
        }

        .slider-block {
          display: flex;
          flex-direction: column;
          gap: 10px;
          color: #e2e8f0;
        }

        .slider-row {
          display: grid;
          grid-template-columns: 1fr auto;
          gap: 12px;
          align-items: center;
        }

        input[type='range'] {
          width: 100%;
          accent-color: #f59e0b;
        }

        .status-strip {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .route-note {
          margin: 0;
          padding: 12px 14px;
          border-radius: 16px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.04);
          line-height: 1.6;
        }

        .route-note.good {
          color: #dcfce7;
          background: rgba(34, 197, 94, 0.12);
          border-color: rgba(34, 197, 94, 0.24);
        }

        .route-note.bad {
          color: #fee2e2;
          background: rgba(239, 68, 68, 0.12);
          border-color: rgba(239, 68, 68, 0.24);
        }

        .route-note.live {
          color: #ffedd5;
          background: rgba(249, 115, 22, 0.12);
          border-color: rgba(249, 115, 22, 0.24);
        }

        .route-note.idle {
          color: rgba(226, 232, 240, 0.82);
        }

        .content-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.7fr);
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
            radial-gradient(circle at 20% 18%, rgba(148, 163, 184, 0.1), transparent 22%),
            linear-gradient(180deg, #f5f7fa 0%, #dce5ee 100%);
        }

        .interactive-canvas {
          cursor: crosshair;
        }

        .legend {
          display: flex;
          gap: 18px;
          flex-wrap: wrap;
          padding: 14px 4px 2px;
        }

        .sidebar {
          display: grid;
          gap: 24px;
        }

        .metric-card {
          padding: 18px;
        }

        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }

        .notes {
          margin: 0;
          padding-left: 18px;
          line-height: 1.6;
        }

        code {
          font-family: "SFMono-Regular", "Menlo", monospace;
          background: rgba(255, 255, 255, 0.08);
          padding: 2px 6px;
          border-radius: 8px;
        }

        @media (max-width: 1100px) {
          .hero-card,
          .content-grid {
            grid-template-columns: 1fr;
          }

          .annotation {
            text-align: left;
            max-width: none;
          }
        }

        @media (max-width: 720px) {
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

          .metric-grid {
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
          background: rgba(34, 197, 94, 0.18);
          color: #dcfce7;
        }

        .bad {
          background: rgba(239, 68, 68, 0.18);
          color: #fee2e2;
        }

        .live {
          background: rgba(249, 115, 22, 0.18);
          color: #ffedd5;
        }

        .idle {
          background: rgba(255, 255, 255, 0.06);
          color: rgba(226, 232, 240, 0.92);
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
          color: rgba(226, 232, 240, 0.88);
          font-size: 0.92rem;
        }

        .swatch {
          width: 14px;
          height: 14px;
          border-radius: 999px;
          box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.18);
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
        }

        span {
          color: rgba(226, 232, 240, 0.72);
          font-size: 0.82rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        strong {
          color: #f8fafc;
          font-size: 1.02rem;
        }
      `}</style>
    </div>
  );
}

function buildScenario(startPoint: Point, goalPoint: Point): Scenario {
  const walls = parseMap(MAP_SOURCE);
  const bounds = getBounds(walls);
  const cellCountX = Math.ceil(bounds.width / GRID_SIZE);
  const cellCountY = Math.ceil(bounds.height / GRID_SIZE);
  const wallCells = Array.from({ length: cellCountY }, () => Array(cellCountX).fill(false));

  for (let j = 0; j < cellCountY; j += 1) {
    for (let i = 0; i < cellCountX; i += 1) {
      const cellMinX = bounds.minX + i * GRID_SIZE;
      const cellMinY = bounds.minY + j * GRID_SIZE;

      wallCells[j][i] = walls.some((wall) => cellContainsWall(wall, cellMinX, cellMinY, GRID_SIZE));
    }
  }

  const blockedCells = inflateOccupiedCells(walls, bounds, wallCells, GRID_SIZE, PLANNER_CLEARANCE);

  const startCell = locateCell(startPoint, bounds, GRID_SIZE, cellCountX, cellCountY);
  const goalCell = locateCell(goalPoint, bounds, GRID_SIZE, cellCountX, cellCountY);

  if (startCell.i === goalCell.i && startCell.j === goalCell.j) {
    throw new Error('Choose distinct start and goal cells.');
  }

  if (wallCells[startCell.j][startCell.i]) {
    throw new Error('Start point lies on a wall.');
  }

  if (wallCells[goalCell.j][goalCell.i]) {
    throw new Error('Goal point lies on a wall.');
  }

  if (distancePointToWalls(startPoint, walls) < PLANNER_CLEARANCE) {
    throw new Error('Start point violates the planner clearance margin.');
  }

  if (distancePointToWalls(goalPoint, walls) < PLANNER_CLEARANCE) {
    throw new Error('Goal point violates the planner clearance margin.');
  }

  const result = runDijkstra({
    blockedCells,
    gridSize: GRID_SIZE,
    cellCountX,
    cellCountY,
    startCell,
    goalCell,
  });

  if (!result) {
    throw new Error('No valid path found for the provided map.');
  }

  const pathCells = prunePath(result, bounds, GRID_SIZE, blockedCells);
  const path = pathCells.map((cell) => cellCenter(cell, bounds, GRID_SIZE));
  path[0] = startPoint;
  path[path.length - 1] = goalPoint;
  const pathArcLengths = buildArcLengths(path);

  return {
    bounds,
    walls,
    wallCells,
    blockedCells,
    gridSize: GRID_SIZE,
    cellCountX,
    cellCountY,
    start: startPoint,
    goal: goalPoint,
    startCell,
    goalCell,
    pathCells,
    path,
    pathArcLengths,
    pathLength: pathArcLengths[pathArcLengths.length - 1] ?? 0,
    planningClearance: PLANNER_CLEARANCE,
  };
}

function createSimulationState(scenario: Scenario): SimulationState {
  const controller = createControllerSnapshot(
    {
      x: scenario.start.x,
      y: scenario.start.y,
      yaw: 0,
      speed: 0,
      steering: 0,
      elapsed: 0,
      reachedGoal: false,
      collided: false,
    },
    scenario,
  );

  return {
    vehicle: {
      x: scenario.start.x,
      y: scenario.start.y,
      yaw: 0,
      speed: 0,
      steering: 0,
      elapsed: 0,
      reachedGoal: false,
      collided: false,
    },
    controller,
    trail: [scenario.start],
    trailCarry: 0,
  };
}

function createTelemetry(simulation: SimulationState, running: boolean): Telemetry {
  const { vehicle, controller } = simulation;
  const front = getSensorReading(controller.sensorReadings, 'front');
  const left = getSensorReading(controller.sensorReadings, 'left');
  const right = getSensorReading(controller.sensorReadings, 'right');

  return {
    x: vehicle.x,
    y: vehicle.y,
    yawDeg: radiansToDegrees(vehicle.yaw),
    speed: vehicle.speed,
    steeringDeg: radiansToDegrees(vehicle.steering),
    elapsed: vehicle.elapsed,
    targetIndex: controller.targetIndex,
    nearestIndex: controller.nearestIndex,
    goalDistance: controller.distanceToGoal,
    running,
    reachedGoal: vehicle.reachedGoal,
    collided: vehicle.collided,
    frontDistance: front.distance,
    leftDistance: left.distance,
    rightDistance: right.distance,
    minClearance: controller.minClearance,
  };
}

function stepSimulation(simulation: SimulationState, scenario: Scenario, dt: number) {
  if (simulation.vehicle.reachedGoal || simulation.vehicle.collided) {
    return;
  }

  const controller = createControllerSnapshot(simulation.vehicle, scenario);
  const steeringTarget = clamp(
    controller.headingError * (1.18 - Math.min(controller.minClearance, 0.5) * 0.36) +
      controller.avoidanceSteering,
    -MAX_STEERING,
    MAX_STEERING,
  );
  const steeringBlend = 1 - Math.exp(-STEERING_RESPONSE * dt);
  simulation.vehicle.steering += (steeringTarget - simulation.vehicle.steering) * steeringBlend;

  const speedError = controller.desiredSpeed - simulation.vehicle.speed;
  const accelLimit = speedError >= 0 ? MAX_ACCELERATION : MAX_DECELERATION;
  const acceleration = clamp(speedError * 2.2, -accelLimit, accelLimit);

  simulation.vehicle.speed = clamp(simulation.vehicle.speed + acceleration * dt, 0, BASE_SPEED);
  simulation.vehicle.x += simulation.vehicle.speed * Math.cos(simulation.vehicle.yaw) * dt;
  simulation.vehicle.y += simulation.vehicle.speed * Math.sin(simulation.vehicle.yaw) * dt;
  simulation.vehicle.yaw = normalizeAngle(
    simulation.vehicle.yaw +
      (simulation.vehicle.speed / WHEELBASE) * Math.tan(simulation.vehicle.steering) * dt,
  );
  simulation.vehicle.elapsed += dt;

  if (isPointInBlockedCell({ x: simulation.vehicle.x, y: simulation.vehicle.y }, scenario)) {
    simulation.vehicle.collided = true;
    simulation.vehicle.speed = 0;
  }

  const updatedController = createControllerSnapshot(simulation.vehicle, scenario);
  simulation.controller = updatedController;

  if (
    updatedController.distanceToGoal < 0.09 &&
    updatedController.targetIndex >= scenario.path.length - 2
  ) {
    simulation.vehicle.reachedGoal = true;
    simulation.vehicle.speed = 0;
  }

  simulation.trailCarry += simulation.vehicle.speed * dt;
  if (simulation.trailCarry >= TRAIL_SPACING) {
    simulation.trailCarry = 0;
    simulation.trail.push({ x: simulation.vehicle.x, y: simulation.vehicle.y });
  }
}

function createControllerSnapshot(vehicle: VehicleState, scenario: Scenario): ControllerSnapshot {
  const nearestSample = findNearestPathSample(
    { x: vehicle.x, y: vehicle.y },
    scenario.path,
    scenario.pathArcLengths,
  );
  const targetSample = samplePathAtProgress(
    scenario.path,
    scenario.pathArcLengths,
    Math.min(nearestSample.progress + LOOKAHEAD_DISTANCE, scenario.pathLength),
  );
  const nearestIndex = nearestSample.segmentIndex;
  const targetIndex = targetSample.segmentIndex + 1;
  const targetPoint = targetSample.point;
  const headingError = normalizeAngle(
    Math.atan2(targetPoint.y - vehicle.y, targetPoint.x - vehicle.x) - vehicle.yaw,
  );
  const distanceToGoal = distanceBetween({ x: vehicle.x, y: vehicle.y }, scenario.goal);
  const sensorReadings = sampleSensors(vehicle, scenario);
  const front = getSensorReading(sensorReadings, 'front');
  const frontLeft = getSensorReading(sensorReadings, 'frontLeft');
  const frontRight = getSensorReading(sensorReadings, 'frontRight');
  const left = getSensorReading(sensorReadings, 'left');
  const right = getSensorReading(sensorReadings, 'right');
  const minClearance = Math.min(...sensorReadings.map((reading) => reading.distance));

  const sideRepulsion = (right.proximity - left.proximity) * 0.72;
  const diagonalRepulsion = (frontRight.proximity - frontLeft.proximity) * 0.92;
  const openSideBias = clamp(
    left.distance + frontLeft.distance * 0.7 - (right.distance + frontRight.distance * 0.7),
    -0.95,
    0.95,
  );
  const frontRepulsion = front.proximity * openSideBias * 0.9;
  const avoidanceSteering = clamp(sideRepulsion + diagonalRepulsion + frontRepulsion, -0.9, 0.9);

  const turnPenalty = clamp(1 - Math.abs(headingError) / 1.15, 0.3, 1);
  const wallSpeedFactor = clamp(
    1 - front.proximity * 0.72 - Math.max(left.proximity, right.proximity) * 0.18,
    0.18,
    1,
  );
  const desiredSpeed =
    distanceToGoal < 0.4
      ? Math.max(distanceToGoal * 1.4, 0)
      : BASE_SPEED * turnPenalty * wallSpeedFactor;

  return {
    nearestIndex,
    targetIndex,
    targetPoint,
    headingError,
    desiredSpeed,
    distanceToGoal,
    sensorReadings,
    avoidanceSteering,
    minClearance,
  };
}

function drawScene(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  map: Scenario,
  simulation: SimulationState,
  pendingStartPoint: Point | null,
) {
  context.clearRect(0, 0, canvas.width, canvas.height);

  const { scale, project } = getCanvasProjection(canvas, map);

  context.fillStyle = '#edf2f7';
  context.fillRect(0, 0, canvas.width, canvas.height);

  drawGrid(context, map, scale, project);
  drawWalls(context, map.walls, project);
  drawPath(context, map.path, project);
  drawTrail(context, simulation.trail, project);
  drawTarget(context, simulation.controller.targetPoint, project);
  drawSensors(context, simulation.controller.sensorReadings, project);
  drawMarkers(context, map.start, map.goal, project, pendingStartPoint);
  drawVehicle(context, simulation.vehicle, project, scale);
}

function getCanvasProjection(canvas: HTMLCanvasElement, map: Scenario) {
  const pad = CANVAS_FRAME_PADDING;
  const scale = Math.min(
    (canvas.width - pad * 2) / map.bounds.width,
    (canvas.height - pad * 2) / map.bounds.height,
  );
  const offsetX = (canvas.width - map.bounds.width * scale) / 2;
  const offsetY = (canvas.height - map.bounds.height * scale) / 2;

  return {
    offsetX,
    offsetY,
    scale,
    project(point: Point): Point {
      return {
        x: offsetX + (point.x - map.bounds.minX) * scale,
        y: canvas.height - offsetY - (point.y - map.bounds.minY) * scale,
      };
    },
    unproject(point: Point): Point {
      return {
        x: map.bounds.minX + (point.x - offsetX) / scale,
        y: map.bounds.minY + (canvas.height - offsetY - point.y) / scale,
      };
    },
  };
}

function getCanvasPixelFromMouseEvent(
  event: MouseEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
): Point {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function canvasToWorldPoint(point: Point, canvas: HTMLCanvasElement, map: Scenario): Point | null {
  const { offsetX, offsetY, scale, unproject } = getCanvasProjection(canvas, map);
  const mapWidth = map.bounds.width * scale;
  const mapHeight = map.bounds.height * scale;
  const withinFrame =
    point.x >= offsetX &&
    point.x <= offsetX + mapWidth &&
    point.y >= canvas.height - offsetY - mapHeight &&
    point.y <= canvas.height - offsetY;

  if (!withinFrame) {
    return null;
  }

  return unproject(point);
}

function drawGrid(
  context: CanvasRenderingContext2D,
  map: Scenario,
  scale: number,
  project: (point: Point) => Point,
) {
  for (let j = 0; j < map.cellCountY; j += 1) {
    for (let i = 0; i < map.cellCountX; i += 1) {
      const min = {
        x: map.bounds.minX + i * map.gridSize,
        y: map.bounds.minY + j * map.gridSize,
      };
      const screen = project({ x: min.x, y: min.y + map.gridSize });
      context.fillStyle = map.wallCells[j][i]
        ? 'rgba(190, 24, 93, 0.22)'
        : map.blockedCells[j][i]
          ? 'rgba(245, 158, 11, 0.14)'
          : 'rgba(148, 163, 184, 0.08)';
      context.strokeStyle = 'rgba(100, 116, 139, 0.13)';
      context.lineWidth = 1;
      context.beginPath();
      context.rect(screen.x, screen.y, map.gridSize * scale, map.gridSize * scale);
      context.fill();
      context.stroke();
    }
  }
}

function drawWalls(
  context: CanvasRenderingContext2D,
  walls: Segment[],
  project: (point: Point) => Point,
) {
  context.strokeStyle = '#0f172a';
  context.lineCap = 'round';
  context.lineWidth = 5.5;

  for (const wall of walls) {
    const start = project({ x: wall.x0, y: wall.y0 });
    const end = project({ x: wall.x1, y: wall.y1 });
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }
}

function drawPath(
  context: CanvasRenderingContext2D,
  path: Point[],
  project: (point: Point) => Point,
) {
  if (path.length < 2) {
    return;
  }

  context.strokeStyle = '#34d399';
  context.lineWidth = 4;
  context.setLineDash([]);
  context.beginPath();
  const first = project(path[0]);
  context.moveTo(first.x, first.y);

  for (let index = 1; index < path.length; index += 1) {
    const waypoint = project(path[index]);
    context.lineTo(waypoint.x, waypoint.y);
  }

  context.stroke();

  for (const waypoint of path) {
    const screen = project(waypoint);
    context.fillStyle = '#065f46';
    context.beginPath();
    context.arc(screen.x, screen.y, 4.6, 0, Math.PI * 2);
    context.fill();
  }
}

function drawTrail(
  context: CanvasRenderingContext2D,
  trail: Point[],
  project: (point: Point) => Point,
) {
  if (trail.length < 2) {
    return;
  }

  context.strokeStyle = 'rgba(244, 211, 94, 0.85)';
  context.lineWidth = 3;
  context.setLineDash([10, 8]);
  context.beginPath();

  const first = project(trail[0]);
  context.moveTo(first.x, first.y);

  for (let index = 1; index < trail.length; index += 1) {
    const point = project(trail[index]);
    context.lineTo(point.x, point.y);
  }

  context.stroke();
  context.setLineDash([]);
}

function drawTarget(
  context: CanvasRenderingContext2D,
  point: Point,
  project: (point: Point) => Point,
) {
  const screen = project(point);
  context.strokeStyle = '#f43f5e';
  context.lineWidth = 2.5;
  context.beginPath();
  context.arc(screen.x, screen.y, 10, 0, Math.PI * 2);
  context.stroke();

  context.beginPath();
  context.moveTo(screen.x - 14, screen.y);
  context.lineTo(screen.x + 14, screen.y);
  context.moveTo(screen.x, screen.y - 14);
  context.lineTo(screen.x, screen.y + 14);
  context.stroke();
}

function drawMarkers(
  context: CanvasRenderingContext2D,
  start: Point,
  goal: Point,
  project: (point: Point) => Point,
  pendingStartPoint: Point | null,
) {
  const startPoint = project(start);
  const goalPoint = project(goal);

  context.fillStyle = '#1d4ed8';
  context.beginPath();
  context.arc(startPoint.x, startPoint.y, 8, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = '#f97316';
  context.beginPath();
  context.arc(goalPoint.x, goalPoint.y, 8, 0, Math.PI * 2);
  context.fill();

  if (pendingStartPoint) {
    const previewPoint = project(pendingStartPoint);

    context.save();
    context.setLineDash([8, 6]);
    context.strokeStyle = 'rgba(59, 130, 246, 0.95)';
    context.lineWidth = 3;
    context.beginPath();
    context.arc(previewPoint.x, previewPoint.y, 16, 0, Math.PI * 2);
    context.stroke();
    context.restore();

    context.fillStyle = '#2563eb';
    context.beginPath();
    context.arc(previewPoint.x, previewPoint.y, 9, 0, Math.PI * 2);
    context.fill();

    context.strokeStyle = '#eff6ff';
    context.lineWidth = 2.5;
    context.beginPath();
    context.arc(previewPoint.x, previewPoint.y, 9, 0, Math.PI * 2);
    context.stroke();
  }
}

function drawSensors(
  context: CanvasRenderingContext2D,
  sensorReadings: SensorReading[],
  project: (point: Point) => Point,
) {
  for (const reading of sensorReadings) {
    const origin = project(reading.origin);
    const hitPoint = project(reading.hitPoint);
    const red = Math.round(239 * reading.proximity + 34 * (1 - reading.proximity));
    const green = Math.round(68 * reading.proximity + 197 * (1 - reading.proximity));

    context.strokeStyle = `rgba(${red}, ${green}, 94, 0.9)`;
    context.lineWidth = reading.id === 'front' ? 3 : 2.2;
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(hitPoint.x, hitPoint.y);
    context.stroke();

    context.fillStyle = `rgba(${red}, ${green}, 94, 0.95)`;
    context.beginPath();
    context.arc(hitPoint.x, hitPoint.y, 4, 0, Math.PI * 2);
    context.fill();
  }
}

function drawVehicle(
  context: CanvasRenderingContext2D,
  vehicle: VehicleState,
  project: (point: Point) => Point,
  scale: number,
) {
  const center = project({ x: vehicle.x, y: vehicle.y });
  const length = VEHICLE_LENGTH * scale;
  const width = VEHICLE_WIDTH * scale;

  context.save();
  context.translate(center.x, center.y);
  context.rotate(-vehicle.yaw);

  context.fillStyle = vehicle.collided ? '#ef4444' : vehicle.reachedGoal ? '#22c55e' : '#fb7185';
  context.beginPath();
  context.rect(-length / 2, -width / 2, length, width);
  context.fill();

  context.fillStyle = '#0f172a';
  context.beginPath();
  context.moveTo(length / 2, 0);
  context.lineTo(length / 2 - 16, -10);
  context.lineTo(length / 2 - 16, 10);
  context.closePath();
  context.fill();

  context.strokeStyle = '#f8fafc';
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(0, 0);
  context.lineTo(length / 2 + 18, 0);
  context.stroke();

  context.restore();
}

function parseMap(source: string): Segment[] {
  return source
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const sanitized = line.endsWith(';') ? line.slice(0, -1) : line;
      const [x0, y0, x1, y1] = sanitized.split(',').map(Number);
      return { x0, y0, x1, y1 };
    });
}

function getBounds(walls: Segment[]): Bounds {
  const xs = walls.flatMap((wall) => [wall.x0, wall.x1]);
  const ys = walls.flatMap((wall) => [wall.y0, wall.y1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

function locateCell(
  point: Point,
  bounds: Bounds,
  gridSize: number,
  cellCountX: number,
  cellCountY: number,
): GridPoint {
  const rawI = Math.floor((point.x - bounds.minX) / gridSize);
  const rawJ = Math.floor((point.y - bounds.minY) / gridSize);

  return {
    i: clamp(rawI, 0, cellCountX - 1),
    j: clamp(rawJ, 0, cellCountY - 1),
  };
}

function cellCenter(cell: GridPoint, bounds: Bounds, gridSize: number): Point {
  return {
    x: bounds.minX + cell.i * gridSize + gridSize / 2,
    y: bounds.minY + cell.j * gridSize + gridSize / 2,
  };
}

function cellContainsWall(wall: Segment, cellMinX: number, cellMinY: number, gridSize: number): boolean {
  const cellMaxX = cellMinX + gridSize;
  const cellMaxY = cellMinY + gridSize;

  if (approximatelyEqual(wall.x0, wall.x1)) {
    const x = wall.x0;
    const wallMinY = Math.min(wall.y0, wall.y1);
    const wallMaxY = Math.max(wall.y0, wall.y1);

    return (
      x >= cellMinX - EPSILON &&
      x <= cellMaxX + EPSILON &&
      rangesOverlap(wallMinY, wallMaxY, cellMinY, cellMaxY)
    );
  }

  if (approximatelyEqual(wall.y0, wall.y1)) {
    const y = wall.y0;
    const wallMinX = Math.min(wall.x0, wall.x1);
    const wallMaxX = Math.max(wall.x0, wall.x1);

    return (
      y >= cellMinY - EPSILON &&
      y <= cellMaxY + EPSILON &&
      rangesOverlap(wallMinX, wallMaxX, cellMinX, cellMaxX)
    );
  }

  return false;
}

function inflateOccupiedCells(
  walls: Segment[],
  bounds: Bounds,
  wallCells: boolean[][],
  gridSize: number,
  clearance: number,
): boolean[][] {
  const cellCountY = wallCells.length;
  const cellCountX = wallCells[0]?.length ?? 0;
  const inflatedCells = Array.from({ length: cellCountY }, () => Array(cellCountX).fill(false));

  for (let j = 0; j < cellCountY; j += 1) {
    for (let i = 0; i < cellCountX; i += 1) {
      if (wallCells[j][i]) {
        inflatedCells[j][i] = true;
        continue;
      }

      const center = cellCenter({ i, j }, bounds, gridSize);
      inflatedCells[j][i] = distancePointToWalls(center, walls) <= clearance;
    }
  }

  return inflatedCells;
}

function runDijkstra({
  blockedCells,
  gridSize,
  cellCountX,
  cellCountY,
  startCell,
  goalCell,
}: {
  blockedCells: boolean[][];
  gridSize: number;
  cellCountX: number;
  cellCountY: number;
  startCell: GridPoint;
  goalCell: GridPoint;
}): GridPoint[] | null {
  const distances = Array.from({ length: cellCountY }, () => Array(cellCountX).fill(Number.POSITIVE_INFINITY));
  const visited = Array.from({ length: cellCountY }, () => Array(cellCountX).fill(false));
  const parents = Array.from({ length: cellCountY }, () => Array<GridPoint | null>(cellCountX).fill(null));
  const frontier: GridPoint[] = [startCell];

  distances[startCell.j][startCell.i] = 0;

  while (frontier.length > 0) {
    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (let index = 0; index < frontier.length; index += 1) {
      const candidate = frontier[index];
      const distance = distances[candidate.j][candidate.i];
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }

    const current = frontier.splice(bestIndex, 1)[0];

    if (visited[current.j][current.i]) {
      continue;
    }

    visited[current.j][current.i] = true;

    if (current.i === goalCell.i && current.j === goalCell.j) {
      break;
    }

    for (const neighbor of NEIGHBORS) {
      const nextI = current.i + neighbor.di;
      const nextJ = current.j + neighbor.dj;

      if (nextI < 0 || nextJ < 0 || nextI >= cellCountX || nextJ >= cellCountY) {
        continue;
      }

      if (blockedCells[nextJ][nextI] || visited[nextJ][nextI]) {
        continue;
      }

      if (
        neighbor.di !== 0 &&
        neighbor.dj !== 0 &&
        (blockedCells[current.j][nextI] || blockedCells[nextJ][current.i])
      ) {
        continue;
      }

      const nextDistance = distances[current.j][current.i] + neighbor.cost * gridSize;

      if (nextDistance < distances[nextJ][nextI]) {
        distances[nextJ][nextI] = nextDistance;
        parents[nextJ][nextI] = current;
        frontier.push({ i: nextI, j: nextJ });
      }
    }
  }

  if (!visited[goalCell.j][goalCell.i]) {
    return null;
  }

  const path: GridPoint[] = [];
  let cursor: GridPoint | null = goalCell;

  while (cursor) {
    path.push(cursor);
    if (cursor.i === startCell.i && cursor.j === startCell.j) {
      break;
    }
    cursor = parents[cursor.j][cursor.i];
  }

  path.reverse();

  if (path.length === 0) {
    return null;
  }

  path[0] = startCell;
  path[path.length - 1] = goalCell;

  return path;
}

function prunePath(
  cells: GridPoint[],
  bounds: Bounds,
  gridSize: number,
  blockedCells: boolean[][],
): GridPoint[] {
  if (cells.length <= 2) {
    return cells;
  }

  const reduced: GridPoint[] = [cells[0]];
  let anchorIndex = 0;

  while (anchorIndex < cells.length - 1) {
    let furthestIndex = anchorIndex + 1;

    for (let probeIndex = cells.length - 1; probeIndex > anchorIndex + 1; probeIndex -= 1) {
      const start = cellCenter(cells[anchorIndex], bounds, gridSize);
      const end = cellCenter(cells[probeIndex], bounds, gridSize);
      if (linePassesFreeSpace(start, end, bounds, gridSize, blockedCells)) {
        furthestIndex = probeIndex;
        break;
      }
    }

    reduced.push(cells[furthestIndex]);
    anchorIndex = furthestIndex;
  }

  return reduced;
}

function linePassesFreeSpace(
  start: Point,
  end: Point,
  bounds: Bounds,
  gridSize: number,
  blockedCells: boolean[][],
): boolean {
  const distance = distanceBetween(start, end);
  const samples = Math.max(Math.ceil(distance / (gridSize * 0.35)), 2);

  for (let index = 0; index <= samples; index += 1) {
    const t = index / samples;
    const point = {
      x: start.x + (end.x - start.x) * t,
      y: start.y + (end.y - start.y) * t,
    };
    const cell = locateCell(point, bounds, gridSize, blockedCells[0].length, blockedCells.length);
    if (blockedCells[cell.j][cell.i]) {
      return false;
    }
  }

  return true;
}

function isPointInBlockedCell(point: Point, scenario: Scenario): boolean {
  if (
    point.x < scenario.bounds.minX ||
    point.x > scenario.bounds.maxX ||
    point.y < scenario.bounds.minY ||
    point.y > scenario.bounds.maxY
  ) {
    return true;
  }

  const cell = locateCell(point, scenario.bounds, scenario.gridSize, scenario.cellCountX, scenario.cellCountY);
  return scenario.wallCells[cell.j][cell.i];
}

function sampleSensors(vehicle: VehicleState, scenario: Scenario): SensorReading[] {
  return SENSOR_CONFIGS.map((config) => {
    const origin = transformLocalPoint(vehicle, { x: config.offsetX, y: config.offsetY });
    const angle = vehicle.yaw + config.relativeAngle;
    const hit = castSensorRay(origin, angle, scenario, config.maxRange);
    const proximity = clamp((config.influenceDistance - hit.distance) / config.influenceDistance, 0, 1);

    return {
      id: config.id,
      label: config.label,
      origin,
      angle,
      distance: hit.distance,
      maxRange: config.maxRange,
      hitPoint: hit.hitPoint,
      proximity,
    };
  });
}

function getSensorReading(sensorReadings: SensorReading[], id: SensorId): SensorReading {
  const reading = sensorReadings.find((candidate) => candidate.id === id);

  if (!reading) {
    throw new Error(`Missing sensor reading for ${id}.`);
  }

  return reading;
}

function transformLocalPoint(vehicle: VehicleState, localPoint: Point): Point {
  const cosYaw = Math.cos(vehicle.yaw);
  const sinYaw = Math.sin(vehicle.yaw);

  return {
    x: vehicle.x + localPoint.x * cosYaw - localPoint.y * sinYaw,
    y: vehicle.y + localPoint.x * sinYaw + localPoint.y * cosYaw,
  };
}

function castSensorRay(
  origin: Point,
  angle: number,
  scenario: Scenario,
  maxRange: number,
): { distance: number; hitPoint: Point } {
  const direction = {
    x: Math.cos(angle),
    y: Math.sin(angle),
  };

  let bestDistance = maxRange;
  let bestPoint = {
    x: origin.x + direction.x * maxRange,
    y: origin.y + direction.y * maxRange,
  };

  for (let j = 0; j < scenario.cellCountY; j += 1) {
    for (let i = 0; i < scenario.cellCountX; i += 1) {
      if (!scenario.blockedCells[j][i]) {
        continue;
      }

      const cellMinX = scenario.bounds.minX + i * scenario.gridSize;
      const cellMinY = scenario.bounds.minY + j * scenario.gridSize;
      const hitDistance = intersectRayWithRect(
        origin,
        direction,
        cellMinX,
        cellMinY,
        scenario.gridSize,
        maxRange,
      );

      if (hitDistance !== null && hitDistance < bestDistance) {
        bestDistance = hitDistance;
        bestPoint = {
          x: origin.x + direction.x * hitDistance,
          y: origin.y + direction.y * hitDistance,
        };
      }
    }
  }

  return {
    distance: bestDistance,
    hitPoint: bestPoint,
  };
}

function intersectRayWithRect(
  origin: Point,
  direction: Point,
  minX: number,
  minY: number,
  size: number,
  maxRange: number,
): number | null {
  const maxX = minX + size;
  const maxY = minY + size;
  let entryDistance = 0;
  let exitDistance = maxRange;

  if (Math.abs(direction.x) <= EPSILON) {
    if (origin.x < minX - EPSILON || origin.x > maxX + EPSILON) {
      return null;
    }
  } else {
    const tx0 = (minX - origin.x) / direction.x;
    const tx1 = (maxX - origin.x) / direction.x;
    entryDistance = Math.max(entryDistance, Math.min(tx0, tx1));
    exitDistance = Math.min(exitDistance, Math.max(tx0, tx1));
  }

  if (Math.abs(direction.y) <= EPSILON) {
    if (origin.y < minY - EPSILON || origin.y > maxY + EPSILON) {
      return null;
    }
  } else {
    const ty0 = (minY - origin.y) / direction.y;
    const ty1 = (maxY - origin.y) / direction.y;
    entryDistance = Math.max(entryDistance, Math.min(ty0, ty1));
    exitDistance = Math.min(exitDistance, Math.max(ty0, ty1));
  }

  if (exitDistance < entryDistance || exitDistance < 0 || entryDistance > maxRange) {
    return null;
  }

  return Math.max(entryDistance, 0);
}

function distancePointToWalls(point: Point, walls: Segment[]): number {
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const wall of walls) {
    bestDistance = Math.min(bestDistance, distancePointToSegment(point, wall));
  }

  return bestDistance;
}

function distancePointToSegment(point: Point, wall: Segment): number {
  const ax = wall.x0;
  const ay = wall.y0;
  const bx = wall.x1;
  const by = wall.y1;
  const abx = bx - ax;
  const aby = by - ay;
  const abLengthSquared = abx * abx + aby * aby;

  if (abLengthSquared <= EPSILON) {
    return Math.hypot(point.x - ax, point.y - ay);
  }

  const apx = point.x - ax;
  const apy = point.y - ay;
  const projection = clamp((apx * abx + apy * aby) / abLengthSquared, 0, 1);
  const closestX = ax + projection * abx;
  const closestY = ay + projection * aby;

  return Math.hypot(point.x - closestX, point.y - closestY);
}

function buildArcLengths(path: Point[]): number[] {
  const arcLengths = [0];

  for (let index = 1; index < path.length; index += 1) {
    arcLengths.push(arcLengths[index - 1] + distanceBetween(path[index - 1], path[index]));
  }

  return arcLengths;
}

function findNearestPathSample(position: Point, path: Point[], arcLengths: number[]) {
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestProgress = 0;
  let bestSegmentIndex = 0;

  for (let index = 0; index < path.length - 1; index += 1) {
    const sample = projectPointToSegment(position, path[index], path[index + 1]);
    if (sample.distance < bestDistance) {
      bestDistance = sample.distance;
      bestProgress = arcLengths[index] + sample.t * distanceBetween(path[index], path[index + 1]);
      bestSegmentIndex = index;
    }
  }

  return {
    progress: bestProgress,
    segmentIndex: bestSegmentIndex,
    distance: bestDistance,
  };
}

function samplePathAtProgress(path: Point[], arcLengths: number[], progress: number) {
  if (path.length === 1) {
    return {
      point: path[0],
      segmentIndex: 0,
    };
  }

  const clampedProgress = clamp(progress, 0, arcLengths[arcLengths.length - 1] ?? 0);

  for (let index = 0; index < arcLengths.length - 1; index += 1) {
    const startProgress = arcLengths[index];
    const endProgress = arcLengths[index + 1];

    if (clampedProgress <= endProgress || index === arcLengths.length - 2) {
      const segmentLength = Math.max(endProgress - startProgress, EPSILON);
      const t = clamp((clampedProgress - startProgress) / segmentLength, 0, 1);
      return {
        point: {
          x: path[index].x + (path[index + 1].x - path[index].x) * t,
          y: path[index].y + (path[index + 1].y - path[index].y) * t,
        },
        segmentIndex: index,
      };
    }
  }

  return {
    point: path[path.length - 1],
    segmentIndex: path.length - 2,
  };
}

function projectPointToSegment(point: Point, start: Point, end: Point) {
  const segment = {
    x: end.x - start.x,
    y: end.y - start.y,
  };
  const segmentLengthSquared = segment.x * segment.x + segment.y * segment.y;

  if (segmentLengthSquared <= EPSILON) {
    return {
      point: start,
      t: 0,
      distance: distanceBetween(point, start),
    };
  }

  const offset = {
    x: point.x - start.x,
    y: point.y - start.y,
  };
  const t = clamp(
    (offset.x * segment.x + offset.y * segment.y) / segmentLengthSquared,
    0,
    1,
  );
  const projectedPoint = {
    x: start.x + segment.x * t,
    y: start.y + segment.y * t,
  };

  return {
    point: projectedPoint,
    t,
    distance: distanceBetween(point, projectedPoint),
  };
}

function distanceBetween(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function normalizeAngle(angle: number): number {
  let normalized = angle;

  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }

  while (normalized < -Math.PI) {
    normalized += Math.PI * 2;
  }

  return normalized;
}

function radiansToDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

function approximatelyEqual(a: number, b: number): boolean {
  return Math.abs(a - b) <= EPSILON;
}

function rangesOverlap(aMin: number, aMax: number, bMin: number, bMax: number): boolean {
  return Math.max(aMin, bMin) <= Math.min(aMax, bMax) + EPSILON;
}
