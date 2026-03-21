'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import DemoShellStyles from './demo-shell-styles';

interface Point {
  x: number;
  y: number;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface House {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface CanvasSize {
  width: number;
  height: number;
}

const BASE_CANVAS_SIZE: CanvasSize = { width: 700, height: 550 };
const CANVAS_PADDING = 20;
const MIN_MAP_X = 1;
const MAX_MAP_X = 11;
const MIN_MAP_Y = 1;
const MAX_MAP_Y = 9;
const DRAWABLE_MAP_WIDTH = MAX_MAP_X - MIN_MAP_X;
const DRAWABLE_MAP_HEIGHT = MAX_MAP_Y - MIN_MAP_Y;
const KNOWN_PARTICLE_COUNT = 10_000;
const UNKNOWN_PARTICLE_COUNT = 80_000;
const ANIMATION_DELAY_MS = 300;
const PROCESSING_DELAY_MS = 100;
const MEASUREMENT_VARIANCE = 0.01;
const MEASUREMENT_LOG_NORMALIZER = -Math.log(2 * Math.PI * MEASUREMENT_VARIANCE);

const ZERO_STATE_MEAN = [0, 0, 0, 0];
const ZERO_MEASUREMENT_MEAN = [0, 0];

const KNOWN_INITIAL_COVARIANCE = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0.5, 0],
  [0, 0, 0, 0.5],
];

const UNKNOWN_PROCESS_COVARIANCE = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0.5, 0],
  [0, 0, 0, 0.5],
];

const DEAD_PARTICLE: Particle = Object.freeze({ x: 0, y: 0, vx: 0, vy: 0 });

const HOUSES: House[] = [
  { x1: 2, y1: 5.2, x2: 4, y2: 8.3 },
  { x1: 2, y1: 3.7, x2: 4, y2: 4.4 },
  { x1: 2, y1: 2, x2: 4, y2: 3.2 },
  { x1: 5, y1: 1, x2: 7, y2: 2.2 },
  { x1: 5, y1: 2.8, x2: 7, y2: 5.5 },
  { x1: 5, y1: 6.2, x2: 7, y2: 9 },
  { x1: 8, y1: 4.6, x2: 10, y2: 8.4 },
  { x1: 8, y1: 2.4, x2: 10, y2: 4 },
  { x1: 8, y1: 1.7, x2: 10, y2: 1.8 },
];

const DEFAULT_STARTING_LOCATION: 'known' | 'unknown' = 'unknown';
const DEFAULT_TRAJECTORY: Point[] = [
  { x: 1.3579121789560895, y: 8.48301574150787 },
  { x: 1.1590720795360399, y: 7.4623032311516155 },
  { x: 1.1855840927920465, y: 6.428334714167357 },
  { x: 1.2783761391880697, y: 5.725766362883181 },
  { x: 1.888152444076222, y: 4.983429991714996 },
  { x: 2.657000828500414, y: 4.797845898922949 },
  { x: 3.757249378624689, y: 4.7580778790389395 },
  { x: 4.433305716652859, y: 4.970173985086992 },
  { x: 4.5658657829328915, y: 5.526926263463132 },
  { x: 5.294946147473074, y: 5.831814415907208 },
  { x: 6.130074565037282, y: 5.818558409279205 },
  { x: 7.031483015741508, y: 5.739022369511185 },
  { x: 7.667771333885667, y: 5.2750621375310685 },
  { x: 8.012427506213754, y: 4.439933719966859 },
  { x: 8.03893951946976, y: 4.028997514498757 },
  { x: 7.813587406793704, y: 3.843413421706711 },
  { x: 7.68102734051367, y: 3.1938690969345487 },
  { x: 7.614747307373654, y: 2.332228666114333 },
  { x: 7.906379453189727, y: 1.4043082021541011 },
  { x: 8.67522783761392, y: 1.152444076222038 },
  { x: 9.788732394366198, y: 1.0463960231980116 },
  { x: 10.862468931234465, y: 1.5368682684341342 },
  { x: 10.822700911350456, y: 2.292460646230323 },
];

const cloneTrajectory = (path: Point[]): Point[] => path.map((point) => ({ ...point }));

const sampleStandardNormalPair = (): [number, number] => {
  let u1 = 0;
  let u2 = 0;

  while (u1 === 0) {
    u1 = Math.random();
  }

  while (u2 === 0) {
    u2 = Math.random();
  }

  const magnitude = Math.sqrt(-2 * Math.log(u1));
  const angle = 2 * Math.PI * u2;

  return [magnitude * Math.cos(angle), magnitude * Math.sin(angle)];
};

const createStandardNormalVector = (size: number): number[] => {
  const vector: number[] = [];

  while (vector.length < size) {
    const [z0, z1] = sampleStandardNormalPair();
    vector.push(z0);

    if (vector.length < size) {
      vector.push(z1);
    }
  }

  return vector;
};

const choleskyDecomposition = (covariance: number[][]): number[][] => {
  const size = covariance.length;
  const lower = Array.from({ length: size }, () => Array(size).fill(0));

  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col <= row; col += 1) {
      let sum = 0;

      for (let k = 0; k < col; k += 1) {
        sum += lower[row][k] * lower[col][k];
      }

      if (row === col) {
        lower[row][col] = Math.sqrt(Math.max(0, covariance[row][row] - sum));
      } else if (lower[col][col] > 0) {
        lower[row][col] = (covariance[row][col] - sum) / lower[col][col];
      }
    }
  }

  return lower;
};

const sampleMultivariateNormal = (mean: number[], cholesky: number[][]): number[] => {
  const standardNormal = createStandardNormalVector(mean.length);

  return mean.map((value, row) => {
    let result = value;

    for (let col = 0; col <= row; col += 1) {
      result += cholesky[row][col] * standardNormal[col];
    }

    return result;
  });
};

const KNOWN_INITIAL_CHOLESKY = choleskyDecomposition(KNOWN_INITIAL_COVARIANCE);
const UNKNOWN_PROCESS_CHOLESKY = choleskyDecomposition(UNKNOWN_PROCESS_COVARIANCE);
const MEASUREMENT_CHOLESKY = choleskyDecomposition([
  [MEASUREMENT_VARIANCE, 0],
  [0, MEASUREMENT_VARIANCE],
]);

const isParticleAlive = (particle: Particle): boolean => particle.x !== 0 || particle.y !== 0;

const isOnRoad = (x: number, y: number): boolean => {
  if (x <= MIN_MAP_X || x >= MAX_MAP_X || y <= MIN_MAP_Y || y >= MAX_MAP_Y) {
    return false;
  }

  for (const house of HOUSES) {
    if (x >= house.x1 && x <= house.x2 && y >= house.y1 && y <= house.y2) {
      return false;
    }
  }

  return true;
};

const sanitizeParticle = (particle: Particle): Particle =>
  isOnRoad(particle.x, particle.y) ? particle : DEAD_PARTICLE;

const createParticleSnapshot = (particleSet: Particle[]): Particle[] =>
  particleSet.map((particle) => ({ ...particle }));

const getAliveParticleCount = (particleSet: Particle[]): number =>
  particleSet.reduce((count, particle) => count + Number(isParticleAlive(particle)), 0);

const buildKnownProcessCholeskies = (path: Point[]): number[][][] =>
  path.map((_, step) => {
    const ddx = step >= 2 ? path[step].x - 2 * path[step - 1].x + path[step - 2].x : 0;
    const ddy = step >= 2 ? path[step].y - 2 * path[step - 1].y + path[step - 2].y : 0;

    return choleskyDecomposition([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, ddx * ddx, 0],
      [0, 0, 0, ddy * ddy],
    ]);
  });

const normalizeWeights = (weights: number[], particleSet: Particle[]): number[] => {
  const sum = weights.reduce((total, weight) => total + weight, 0);

  if (sum > 0) {
    return weights.map((weight) => weight / sum);
  }

  const aliveCount = getAliveParticleCount(particleSet);
  if (aliveCount === 0) {
    return weights.map(() => 0);
  }

  const fallbackWeight = 1 / aliveCount;
  return particleSet.map((particle) => (isParticleAlive(particle) ? fallbackWeight : 0));
};

const systematicResample = (particleSet: Particle[], weights: number[]): Particle[] => {
  if (particleSet.length === 0) {
    return [];
  }

  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  if (totalWeight <= 0) {
    return createParticleSnapshot(particleSet);
  }

  const step = 1 / particleSet.length;
  let threshold = Math.random() * step;
  let cumulativeWeight = weights[0];
  let sourceIndex = 0;
  const resampled = new Array<Particle>(particleSet.length);

  for (let index = 0; index < particleSet.length; index += 1) {
    while (threshold > cumulativeWeight && sourceIndex < particleSet.length - 1) {
      sourceIndex += 1;
      cumulativeWeight += weights[sourceIndex];
    }

    resampled[index] = { ...particleSet[sourceIndex] };
    threshold += step;
  }

  return resampled;
};

const ParticleFilterMap: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);
  const processingTimeoutRef = useRef<number | null>(null);

  const [startingLocation, setStartingLocation] =
    useState<'known' | 'unknown'>(DEFAULT_STARTING_LOCATION);
  const [trajectory, setTrajectory] = useState<Point[]>(() => cloneTrajectory(DEFAULT_TRAJECTORY));
  const [isDrawing, setIsDrawing] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [particles, setParticles] = useState<Particle[][]>([]);
  const [activeParticleCounts, setActiveParticleCounts] = useState<number[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>(BASE_CANVAS_SIZE);
  const [isProcessing, setIsProcessing] = useState(false);

  const totalParticleCount =
    startingLocation === 'known' ? KNOWN_PARTICLE_COUNT : UNKNOWN_PARTICLE_COUNT;
  const pointLabel = trajectory.length === 1 ? 'point' : 'points';
  const activeParticleCount = activeParticleCounts[currentFrame] ?? 0;
  const baseButtonClass =
    'inline-flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50';

  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) {
        return;
      }

      const containerWidth = containerRef.current.clientWidth;
      const usableWidth = Math.max(320, Math.min(containerWidth - 40, 900));
      const scale = usableWidth / BASE_CANVAS_SIZE.width;

      setCanvasSize({
        width: BASE_CANVAS_SIZE.width * scale,
        height: BASE_CANVAS_SIZE.height * scale,
      });
    };

    updateSize();
    window.addEventListener('resize', updateSize);

    return () => {
      window.removeEventListener('resize', updateSize);

      if (animationRef.current !== null) {
        window.clearTimeout(animationRef.current);
      }

      if (processingTimeoutRef.current !== null) {
        window.clearTimeout(processingTimeoutRef.current);
      }
    };
  }, []);

  const getScale = useCallback(
    () =>
      Math.min(
        (canvasSize.width - CANVAS_PADDING * 2) / DRAWABLE_MAP_WIDTH,
        (canvasSize.height - CANVAS_PADDING * 2) / DRAWABLE_MAP_HEIGHT,
      ),
    [canvasSize.height, canvasSize.width],
  );

  const toCanvas = useCallback((x: number, y: number): Point => {
    const scale = getScale();

    return {
      x: (x - MIN_MAP_X) * scale + CANVAS_PADDING,
      y: canvasSize.height - ((y - MIN_MAP_Y) * scale + CANVAS_PADDING),
    };
  }, [canvasSize.height, getScale]);

  const fromCanvas = useCallback((canvasX: number, canvasY: number): Point => {
    const scale = getScale();

    return {
      x: (canvasX - CANVAS_PADDING) / scale + MIN_MAP_X,
      y: (canvasSize.height - canvasY - CANVAS_PADDING) / scale + MIN_MAP_Y,
    };
  }, [canvasSize.height, getScale]);

  const drawMap = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
    ctx.fillStyle = '#06111b';
    ctx.fillRect(0, 0, canvasSize.width, canvasSize.height);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;

    const boundarySegments = [
      [
        { x: 1, y: 1 },
        { x: 1, y: 9 },
        { x: 5, y: 9 },
      ],
      [
        { x: 7, y: 9 },
        { x: 11, y: 9 },
        { x: 11, y: 1 },
        { x: 7, y: 1 },
      ],
      [
        { x: 5, y: 1 },
        { x: 1, y: 1 },
      ],
    ];

    boundarySegments.forEach((segment) => {
      ctx.beginPath();
      const first = toCanvas(segment[0].x, segment[0].y);
      ctx.moveTo(first.x, first.y);

      for (let index = 1; index < segment.length; index += 1) {
        const point = toCanvas(segment[index].x, segment[index].y);
        ctx.lineTo(point.x, point.y);
      }

      ctx.stroke();
    });

    ctx.fillStyle = '#102437';
    ctx.strokeStyle = '#22d3ee';
    HOUSES.forEach((house) => {
      const p1 = toCanvas(house.x1, house.y1);
      const p2 = toCanvas(house.x2, house.y2);
      ctx.fillRect(p1.x, p2.y, p2.x - p1.x, p1.y - p2.y);
      ctx.strokeRect(p1.x, p2.y, p2.x - p1.x, p1.y - p2.y);
    });
  }, [canvasSize.height, canvasSize.width, toCanvas]);

  const drawTrajectory = useCallback((ctx: CanvasRenderingContext2D, upToFrame?: number) => {
    if (trajectory.length === 0) {
      return;
    }

    const maxPoint =
      upToFrame !== undefined ? Math.min(upToFrame, trajectory.length - 1) : trajectory.length - 1;

    if (maxPoint >= 1) {
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 3;
      ctx.beginPath();
      const first = toCanvas(trajectory[0].x, trajectory[0].y);
      ctx.moveTo(first.x, first.y);

      for (let index = 1; index <= maxPoint; index += 1) {
        const point = toCanvas(trajectory[index].x, trajectory[index].y);
        ctx.lineTo(point.x, point.y);
      }

      ctx.stroke();
    }

    trajectory.forEach((point, index) => {
      const canvasPoint = toCanvas(point.x, point.y);
      if (index <= maxPoint) {
        ctx.fillStyle = index === maxPoint ? '#10b981' : '#ef4444';
      } else {
        ctx.fillStyle = '#475569';
      }

      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, index === maxPoint ? 8 : 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(index + 1), canvasPoint.x, canvasPoint.y);
    });
  }, [toCanvas, trajectory]);

  const drawParticles = useCallback((ctx: CanvasRenderingContext2D, particleSet: Particle[]) => {
    ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';

    particleSet.forEach((particle) => {
      if (!isParticleAlive(particle)) {
        return;
      }

      const canvasPoint = toCanvas(particle.x, particle.y);
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [toCanvas]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    drawMap(ctx);

    if (particles.length > 0 && currentFrame < particles.length) {
      drawParticles(ctx, particles[currentFrame]);
      drawTrajectory(ctx, currentFrame);
      return;
    }

    drawTrajectory(ctx);
  }, [canvasSize, currentFrame, drawMap, drawParticles, drawTrajectory, particles, trajectory]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;
    const point = fromCanvas(canvasX, canvasY);

    if (isOnRoad(point.x, point.y)) {
      setTrajectory((previous) => [...previous, point]);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, 10, 0, Math.PI * 2);
    ctx.fill();

    window.setTimeout(() => {
      drawMap(ctx);
      drawTrajectory(ctx);
    }, 200);
  };

  const runParticleFilter = () => {
    if (trajectory.length < 2) {
      window.alert('Please create a trajectory with at least 2 points.');
      return;
    }

    if (processingTimeoutRef.current !== null) {
      window.clearTimeout(processingTimeoutRef.current);
    }

    setIsDrawing(false);
    setIsAnimating(false);
    setIsPaused(false);
    setIsProcessing(true);
    setCurrentFrame(0);

    processingTimeoutRef.current = window.setTimeout(() => {
      try {
        const particleCount =
          startingLocation === 'known' ? KNOWN_PARTICLE_COUNT : UNKNOWN_PARTICLE_COUNT;
        const processCholeskies =
          startingLocation === 'known'
            ? buildKnownProcessCholeskies(trajectory)
            : trajectory.map(() => UNKNOWN_PROCESS_CHOLESKY);

        const particleFrames: Particle[][] = [];
        const frameActiveCounts: number[] = [];

        let currentParticles: Particle[];

        if (startingLocation === 'known') {
          const initialState = [trajectory[0].x, trajectory[0].y, 0, 0];

          currentParticles = Array.from({ length: particleCount }, () => {
            const sample = sampleMultivariateNormal(initialState, KNOWN_INITIAL_CHOLESKY);
            return {
              x: sample[0],
              y: sample[1],
              vx: sample[2],
              vy: sample[3],
            };
          });
        } else {
          currentParticles = Array.from({ length: particleCount }, () => ({
            x: MIN_MAP_X + Math.random() * (MAX_MAP_X - MIN_MAP_X),
            y: MIN_MAP_Y + Math.random() * (MAX_MAP_Y - MIN_MAP_Y),
            vx: -0.1 + Math.random() * 0.2,
            vy: -0.1 + Math.random() * 0.2,
          }));
        }

        currentParticles = currentParticles.map(sanitizeParticle);
        particleFrames.push(createParticleSnapshot(currentParticles));
        frameActiveCounts.push(getAliveParticleCount(currentParticles));

        let weights = Array.from({ length: particleCount }, () => 1 / particleCount);

        for (let step = 1; step < trajectory.length; step += 1) {
          const actualVelocity = {
            x: trajectory[step].x - trajectory[step - 1].x,
            y: trajectory[step].y - trajectory[step - 1].y,
          };

          const measurementNoise = sampleMultivariateNormal(
            ZERO_MEASUREMENT_MEAN,
            MEASUREMENT_CHOLESKY,
          );
          const measuredVelocity = {
            x: actualVelocity.x + measurementNoise[0],
            y: actualVelocity.y + measurementNoise[1],
          };

          const processNoiseCholesky = processCholeskies[step];

          currentParticles = currentParticles.map((particle) => {
            const noise = sampleMultivariateNormal(ZERO_STATE_MEAN, processNoiseCholesky);

            return sanitizeParticle({
              x: particle.x + particle.vx + noise[0],
              y: particle.y + particle.vy + noise[1],
              vx: particle.vx + noise[2],
              vy: particle.vy + noise[3],
            });
          });

          const logWeights = currentParticles.map((particle, index) => {
            if (!isParticleAlive(particle)) {
              return Number.NEGATIVE_INFINITY;
            }

            const diffX = particle.vx - measuredVelocity.x;
            const diffY = particle.vy - measuredVelocity.y;
            const mahalanobis = (diffX * diffX + diffY * diffY) / MEASUREMENT_VARIANCE;

            return Math.log(weights[index]) + MEASUREMENT_LOG_NORMALIZER - 0.5 * mahalanobis;
          });

          const maxLogWeight = Math.max(...logWeights);
          const unnormalizedWeights = Number.isFinite(maxLogWeight)
            ? logWeights.map((logWeight) =>
                Number.isFinite(logWeight) ? Math.exp(logWeight - maxLogWeight) : 0,
              )
            : currentParticles.map((particle) => (isParticleAlive(particle) ? 1 : 0));

          weights = normalizeWeights(unnormalizedWeights, currentParticles);
          currentParticles = systematicResample(currentParticles, weights);
          weights = Array.from({ length: particleCount }, () => 1 / particleCount);

          particleFrames.push(createParticleSnapshot(currentParticles));
          frameActiveCounts.push(getAliveParticleCount(currentParticles));
        }

        setParticles(particleFrames);
        setActiveParticleCounts(frameActiveCounts);
        setIsProcessing(false);
        setIsAnimating(true);
        setIsPaused(false);
      } catch (error) {
        console.error('Particle filter failed:', error);
        setIsDrawing(true);
        setIsProcessing(false);
      } finally {
        processingTimeoutRef.current = null;
      }
    }, PROCESSING_DELAY_MS);
  };

  useEffect(() => {
    if (!isAnimating || isPaused || particles.length === 0) {
      return undefined;
    }

    if (currentFrame >= particles.length - 1) {
      setIsAnimating(false);
      return undefined;
    }

    animationRef.current = window.setTimeout(() => {
      setCurrentFrame((previous) => previous + 1);
    }, ANIMATION_DELAY_MS);

    return () => {
      if (animationRef.current !== null) {
        window.clearTimeout(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [currentFrame, isAnimating, isPaused, particles.length]);

  const reset = (mode: 'default' | 'empty' = 'default') => {
    if (animationRef.current !== null) {
      window.clearTimeout(animationRef.current);
      animationRef.current = null;
    }

    if (processingTimeoutRef.current !== null) {
      window.clearTimeout(processingTimeoutRef.current);
      processingTimeoutRef.current = null;
    }

    setStartingLocation(DEFAULT_STARTING_LOCATION);
    setTrajectory(mode === 'default' ? cloneTrajectory(DEFAULT_TRAJECTORY) : []);
    setParticles([]);
    setActiveParticleCounts([]);
    setCurrentFrame(0);
    setIsDrawing(true);
    setIsAnimating(false);
    setIsPaused(false);
    setIsProcessing(false);
  };

  return (
    <main className="page-shell demo-shell">
      <DemoShellStyles />

      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Bayesian Tracking / Interactive Map</p>
          <h1>Particle Filter Map</h1>
          <p className="lede">
            Sketch a vehicle trajectory across the road network, choose whether the start is known
            or unknown, and replay how the particle cloud converges on the route over time.
          </p>
        </div>

        <div className="control-panel">
          <div>
            <label
              className="mini-label"
              htmlFor="starting-location"
            >
              Starting Location
            </label>
            <div className="mt-2">
              <select
                id="starting-location"
                value={startingLocation}
                onChange={(event) =>
                  setStartingLocation(event.target.value as 'known' | 'unknown')
                }
                disabled={!isDrawing || trajectory.length > 0}
                className="w-full rounded-2xl border border-slate-700 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="known">Known</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>
          </div>

          <div className="button-row">
            {isDrawing && trajectory.length >= 2 && (
              <button
                type="button"
                onClick={runParticleFilter}
                disabled={isProcessing}
                className={`${baseButtonClass} border-cyan-400/30 bg-cyan-400/15 text-cyan-100 hover:border-cyan-300/60 hover:bg-cyan-400/20`}
              >
                <Play size={18} />
                {isProcessing ? 'Processing...' : 'Run Filter'}
              </button>
            )}

            {!isDrawing && !isProcessing && particles.length > 0 && (
              <>
                {!isAnimating || isPaused ? (
                  <button
                    type="button"
                    onClick={() => {
                      if (currentFrame >= particles.length - 1) {
                        setCurrentFrame(0);
                      }

                      setIsAnimating(true);
                      setIsPaused(false);
                    }}
                    className={`${baseButtonClass} border-emerald-400/30 bg-emerald-400/15 text-emerald-100 hover:border-emerald-300/60 hover:bg-emerald-400/20`}
                  >
                    <Play size={18} />
                    {currentFrame >= particles.length - 1 ? 'Replay' : 'Play'}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setIsPaused(true)}
                    className={`${baseButtonClass} border-amber-400/30 bg-amber-400/15 text-amber-100 hover:border-amber-300/60 hover:bg-amber-400/20`}
                  >
                    <Pause size={18} />
                    Pause
                  </button>
                )}
              </>
            )}

            <button
              type="button"
              onClick={() => reset('empty')}
              disabled={isProcessing}
              className={`${baseButtonClass} border-slate-700 bg-slate-950/70 text-slate-100 hover:border-slate-600 hover:bg-slate-900`}
            >
              Set new path
            </button>

            <button
              type="button"
              onClick={() => reset('default')}
              disabled={isProcessing}
              className={`${baseButtonClass} border-slate-700 bg-slate-950/70 text-slate-100 hover:border-slate-600 hover:bg-slate-900`}
            >
              <RotateCcw size={18} />
              Reset to default
            </button>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <article className="canvas-card">
          <div className="section-head">
            <div>
              <p className="mini-label">Tracking Workspace</p>
              <h2>Road map and particle cloud</h2>
            </div>
            <p className="annotation">
              {trajectory.length} {pointLabel} recorded.{' '}
              {startingLocation === 'known'
                ? 'Known starts initialize near the first waypoint.'
                : 'Unknown starts spread particles across the drivable map.'}
            </p>
          </div>

          {isProcessing && (
            <div className="surface-inset mb-4 p-4">
              <div className="flex items-center gap-3">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-cyan-200/30 border-b-cyan-200" />
                <span className="text-sm font-medium text-cyan-50">
                  Running particle filter with {totalParticleCount.toLocaleString()} particles...
                </span>
              </div>
            </div>
          )}

          {particles.length > 0 && !isProcessing && (
            <div className="surface-inset mb-4 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <span className="text-sm font-medium text-slate-300">Frame</span>
                <input
                  type="range"
                  min="0"
                  max={particles.length - 1}
                  value={currentFrame}
                  onChange={(event) => {
                    setCurrentFrame(Number.parseInt(event.target.value, 10));
                    if (isAnimating) {
                      setIsPaused(true);
                    }
                  }}
                  className="flex-1 accent-cyan-400"
                />
                <span className="min-w-[72px] text-right text-sm text-slate-300">
                  {currentFrame + 1} / {particles.length}
                </span>
              </div>
              <div className="mt-3 text-xs uppercase tracking-[0.18em] text-slate-400">
                Active particles: {activeParticleCount.toLocaleString()} /{' '}
                {totalParticleCount.toLocaleString()}
              </div>
            </div>
          )}

          <div
            ref={containerRef}
            className="viz-surface"
          >
            <canvas
              ref={canvasRef}
              width={canvasSize.width}
              height={canvasSize.height}
              onClick={handleCanvasClick}
              className={`block w-full ${isDrawing ? 'cursor-crosshair' : 'cursor-default'}`}
            />
          </div>

          <div className="legend-strip mt-4 text-sm">
            <span className="inline-flex items-center gap-2 text-cyan-100">
              <span className="h-3 w-3 rounded-full bg-cyan-300" />
              Buildings / blocked regions
            </span>
            <span className="inline-flex items-center gap-2 text-rose-100">
              <span className="h-3 w-3 rounded-full bg-rose-400" />
              Particles
            </span>
            <span className="inline-flex items-center gap-2 text-slate-200">
              <span className="h-3 w-3 rounded-full bg-slate-300" />
              Waypoints
            </span>
            <span className="inline-flex items-center gap-2 text-emerald-100">
              <span className="h-3 w-3 rounded-full bg-emerald-400" />
              Current point
            </span>
          </div>
        </article>

        <aside className="sidebar">
          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Workflow</p>
                <h2>How to use it</h2>
              </div>
            </div>
            <ul className="notes-list">
              <li>A default unknown-start trajectory is preloaded when the page opens.</li>
              <li>Use <code>Set new path</code> to clear the preset and sketch a route from scratch.</li>
              <li>Click on the dark road network to extend the route with extra waypoints if needed.</li>
              <li>Run the filter, then scrub or replay frames to inspect convergence.</li>
            </ul>
          </article>

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Tracking Notes</p>
                <h2>What you are seeing</h2>
              </div>
            </div>
            <p className="detail-copy">
              The filter updates particle velocity using noisy motion observations derived from the
              hand-drawn path. Known starts converge faster because the initial particle cloud is
              seeded close to the first waypoint.
            </p>
            <p className="detail-copy subtle mt-3">
              Unknown starts intentionally begin much wider so you can compare the cost of initial
              uncertainty against the cleaner known-start case.
            </p>
          </article>
        </aside>
      </section>
    </main>
  );
};

export default ParticleFilterMap;
