'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';
import DemoShellStyles from './demo-shell-styles';

type Vector = number[];
type Matrix = number[][];

interface GaussianState {
  x: Vector;
  P: Matrix;
}

interface Bernoulli {
  r: number;
  state: GaussianState;
  t_birth: number;
  t_death: number[];
  w_death: number[];
}

interface PMBMParams {
  PPP: {
    w: number[];
    states: GaussianState[];
  };
  MBM: {
    w: number[];
    ht: number[][];
    tt: Bernoulli[][];
  };
}

interface AssignmentSolution {
  assignment: number[];
  cost: number;
}

interface FinancialEstimate {
  price: number;
  velocity: number;
  existence: number;
  variance: number;
}

const seededRandomNormal = (rng: () => number): number => {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const hashString = (value: string): number => {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const createSeededRandom = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const matrixMultiply = (A: number[][], B: number[][]): number[][] => {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const result = Array.from({ length: rowsA }, () => Array(colsB).fill(0));
  
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
};

const matrixTranspose = (A: number[][]): number[][] => {
  return A[0].map((_, i) => A.map(row => row[i]));
};

const matrixAdd = (A: Matrix, B: Matrix): Matrix =>
  A.map((row, i) => row.map((value, j) => value + B[i][j]));

const vectorToColumn = (v: Vector): Matrix => v.map(value => [value]);

const columnToVector = (m: Matrix): Vector => m.map(row => row[0]);

const matrixInverse = (A: number[][]): number[][] => {
  const n = A.length;
  const augmented = A.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);
  
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
    
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k][i] / augmented[i][i];
      for (let j = i; j < 2 * n; j++) {
        augmented[k][j] -= factor * augmented[i][j];
      }
    }
  }
  
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i - 1; k >= 0; k--) {
      const factor = augmented[k][i] / augmented[i][i];
      for (let j = 0; j < 2 * n; j++) {
        augmented[k][j] -= factor * augmented[i][j];
      }
    }
  }
  
  for (let i = 0; i < n; i++) {
    const divisor = augmented[i][i];
    for (let j = 0; j < 2 * n; j++) {
      augmented[i][j] /= divisor;
    }
  }
  
  return augmented.map(row => row.slice(n));
};

const normalizeLogWeights = (logWeights: number[]): [number[], number] => {
  if (logWeights.length === 0) return [[], -Infinity];
  if (logWeights.length === 1) return [[0], logWeights[0]];

  const maxLog = Math.max(...logWeights);
  const logSum = maxLog + Math.log(logWeights.reduce((sum, weight) => sum + Math.exp(weight - maxLog), 0));
  return [logWeights.map(weight => weight - logSum), logSum];
};

const logSumExp = (logWeights: number[]): number => normalizeLogWeights(logWeights)[1];

const stableUnique = (values: number[]): number[] => {
  const seen = new Set<number>();
  const unique: number[] = [];
  values.forEach(value => {
    if (!seen.has(value)) {
      seen.add(value);
      unique.push(value);
    }
  });
  return unique;
};

const roundDownToStep = (value: number, step: number): number => {
  if (step <= 0) return value;
  return Math.floor(value / step) * step;
};

const insertAssignmentSolution = (
  solutions: AssignmentSolution[],
  candidate: AssignmentSolution,
  limit: number
) => {
  let insertAt = solutions.findIndex(solution => candidate.cost < solution.cost);
  if (insertAt < 0) insertAt = solutions.length;
  solutions.splice(insertAt, 0, candidate);
  if (solutions.length > limit) {
    solutions.length = limit;
  }
};

const kBestAssignments = (costMatrix: number[][], k: number): AssignmentSolution[] => {
  const rowCount = costMatrix.length;
  if (rowCount === 0) return [{ assignment: [], cost: 0 }];
  if (k <= 0) return [];

  const colCount = costMatrix[0].length;
  const existingTrackCols = Math.max(0, colCount - rowCount);
  const optionsByRow = Array.from({ length: rowCount }, (_, rowIdx) => {
    const options: { col: number; cost: number }[] = [];
    for (let colIdx = 0; colIdx < existingTrackCols; colIdx++) {
      const cost = costMatrix[rowIdx][colIdx];
      if (Number.isFinite(cost)) {
        options.push({ col: colIdx, cost });
      }
    }
    const fallbackCol = existingTrackCols + rowIdx;
    if (fallbackCol < colCount && Number.isFinite(costMatrix[rowIdx][fallbackCol])) {
      options.push({ col: fallbackCol, cost: costMatrix[rowIdx][fallbackCol] });
    }
    options.sort((a, b) => a.cost - b.cost);
    return options;
  });

  const rowOrder = Array.from({ length: rowCount }, (_, rowIdx) => rowIdx).sort((a, b) => {
    const countDiff = optionsByRow[a].length - optionsByRow[b].length;
    if (countDiff !== 0) return countDiff;
    return optionsByRow[a][0].cost - optionsByRow[b][0].cost;
  });

  const orderedOptions = rowOrder.map(rowIdx => optionsByRow[rowIdx]);
  const usedExistingCols = Array(existingTrackCols).fill(false);
  const assignment = Array(rowCount).fill(-1);
  const solutions: AssignmentSolution[] = [];

  const lowerBound = (startDepth: number): number => {
    let bound = 0;
    for (let depth = startDepth; depth < orderedOptions.length; depth++) {
      const bestOption = orderedOptions[depth].find(
        option => option.col >= existingTrackCols || !usedExistingCols[option.col]
      );
      if (!bestOption) return Infinity;
      bound += bestOption.cost;
    }
    return bound;
  };

  const search = (depth: number, currentCost: number) => {
    const optimisticCost = currentCost + lowerBound(depth);
    const cutoff = solutions.length === k ? solutions[solutions.length - 1].cost : Infinity;
    if (!Number.isFinite(optimisticCost) || optimisticCost > cutoff) {
      return;
    }

    if (depth === rowCount) {
      insertAssignmentSolution(solutions, { assignment: [...assignment], cost: currentCost }, k);
      return;
    }

    const rowIdx = rowOrder[depth];
    for (const option of orderedOptions[depth]) {
      if (option.col < existingTrackCols && usedExistingCols[option.col]) continue;

      if (option.col < existingTrackCols) {
        usedExistingCols[option.col] = true;
      }
      assignment[rowIdx] = option.col;
      search(depth + 1, currentCost + option.cost);
      assignment[rowIdx] = -1;
      if (option.col < existingTrackCols) {
        usedExistingCols[option.col] = false;
      }
    }
  };

  search(0, 0);
  return solutions;
};

// Linear-Gaussian prediction primitive used inside the PMBM filter
const kalmanPredict = (
  x: number[], 
  P: number[][], 
  F: number[][], 
  Q: number[][]
): { x: number[], P: number[][] } => {
  const xPred = [
    F[0][0] * x[0] + F[0][1] * x[1],
    F[1][0] * x[0] + F[1][1] * x[1]
  ];
  
  const FP = matrixMultiply(F, P);
  const FPFt = matrixMultiply(FP, matrixTranspose(F));
  const PPred = FPFt.map((row, i) => row.map((val, j) => val + Q[i][j]));
  
  return { x: xPred, P: PPred };
};

// Linear-Gaussian update primitive used inside the PMBM filter
const kalmanUpdate = (
  x: number[], 
  P: number[][], 
  y: number[], 
  H: number[][], 
  R: number[][]
): { x: number[], P: number[][] } => {
  const HP = matrixMultiply(H, P);
  const HPHt = matrixMultiply(HP, matrixTranspose(H));
  const S = HPHt.map((row, i) => row.map((val, j) => val + R[i][j]));
  
  const SInv = matrixInverse(S);
  const K = matrixMultiply(matrixMultiply(P, matrixTranspose(H)), SInv);
  
  const hx = [x[0]];
  const innovation = y.map((val, i) => val - hx[i]);
  const xUpd = x.map((val, i) => val + K[i].reduce((sum, kVal, j) => sum + kVal * innovation[j], 0));
  
  const KH = matrixMultiply(K, H);
  const I = [[1, 0], [0, 1]];
  const IKH = I.map((row, i) => row.map((val, j) => val - KH[i][j]));
  const PUpd = matrixMultiply(IKH, P);
  
  return { x: xUpd, P: PUpd };
};

class FinancialGaussianDensity {
  static predict(state: GaussianState, F: Matrix, Q: Matrix): GaussianState {
    const prediction = kalmanPredict(state.x, state.P, F, Q);
    return { x: prediction.x, P: prediction.P };
  }

  static update(state: GaussianState, measurement: Vector, H: Matrix, R: Matrix): GaussianState {
    const updated = kalmanUpdate(state.x, state.P, measurement, H, R);
    return { x: updated.x, P: updated.P };
  }

  static predictedLikelihood(state: GaussianState, measurement: Vector, H: Matrix, R: Matrix): number {
    const xCol = vectorToColumn(state.x);
    const zPred = columnToVector(matrixMultiply(H, xCol));
    const HP = matrixMultiply(H, state.P);
    const S = matrixAdd(matrixMultiply(HP, matrixTranspose(H)), R);
    const variance = Math.max(S[0][0], 1e-9);
    const innovation = measurement[0] - zPred[0];
    return -0.5 * ((innovation * innovation) / variance + Math.log(2 * Math.PI * variance));
  }

  static ellipsoidalGating(
    state: GaussianState,
    measurements: Vector[],
    H: Matrix,
    R: Matrix,
    gatingSize: number
  ): boolean[] {
    if (measurements.length === 0) return [];
    const xCol = vectorToColumn(state.x);
    const zPred = columnToVector(matrixMultiply(H, xCol));
    const HP = matrixMultiply(H, state.P);
    const S = matrixAdd(matrixMultiply(HP, matrixTranspose(H)), R);
    const variance = Math.max(S[0][0], 1e-9);
    return measurements.map(measurement => {
      const innovation = measurement[0] - zPred[0];
      return (innovation * innovation) / variance < gatingSize;
    });
  }

  static momentMatching(logWeights: number[], states: GaussianState[]): GaussianState {
    if (states.length === 1) {
      return {
        x: [...states[0].x],
        P: states[0].P.map(row => [...row])
      };
    }

    const [normalizedWeights] = normalizeLogWeights(logWeights);
    const weights = normalizedWeights.map(weight => Math.exp(weight));
    const stateDim = states[0].x.length;

    const x = Array.from({ length: stateDim }, (_, dim) =>
      states.reduce((sum, state, idx) => sum + weights[idx] * state.x[dim], 0)
    );

    const P = Array.from({ length: stateDim }, () => Array(stateDim).fill(0));
    for (let idx = 0; idx < states.length; idx++) {
      const diff = states[idx].x.map((value, dim) => value - x[dim]);
      for (let row = 0; row < stateDim; row++) {
        for (let col = 0; col < stateDim; col++) {
          P[row][col] += weights[idx] * (states[idx].P[row][col] + diff[row] * diff[col]);
        }
      }
    }

    return { x, P };
  }
}

class FinancialPMBMFilter {
  params: PMBMParams;

  constructor() {
    this.params = {
      PPP: { w: [], states: [] },
      MBM: { w: [], ht: [], tt: [] }
    };
  }

  private cloneState(state: GaussianState): GaussianState {
    return {
      x: [...state.x],
      P: state.P.map(row => [...row])
    };
  }

  private cloneBernoulli(bern: Bernoulli): Bernoulli {
    return {
      r: bern.r,
      state: this.cloneState(bern.state),
      t_birth: bern.t_birth,
      t_death: [...bern.t_death],
      w_death: [...bern.w_death]
    };
  }

  private bernoulliUndetectedUpdate(bern: Bernoulli, P_D: number): { bern: Bernoulli; logLik: number } {
    const updated = this.cloneBernoulli(bern);
    const aliveWeight = bern.w_death[bern.w_death.length - 1];
    const lNoDetect = bern.r * (1 - P_D * aliveWeight);
    const likUndetected = 1 - bern.r + lNoDetect;
    const normalizer = Math.max(1 - aliveWeight * P_D, 1e-9);

    updated.r = likUndetected > 0 ? lNoDetect / likUndetected : 0;
    updated.w_death = [
      ...bern.w_death.slice(0, -1),
      aliveWeight * (1 - P_D)
    ].map(weight => weight / normalizer);

    return { bern: updated, logLik: Math.log(Math.max(likUndetected, 1e-12)) };
  }

  private bernoulliDetectedLikelihood(
    bern: Bernoulli,
    measurement: Vector,
    H: Matrix,
    R: Matrix,
    P_D: number
  ): number {
    const aliveWeight = bern.w_death[bern.w_death.length - 1];
    return (
      FinancialGaussianDensity.predictedLikelihood(bern.state, measurement, H, R) +
      Math.log(Math.max(P_D * bern.r * aliveWeight, 1e-12))
    );
  }

  private bernoulliDetectedUpdateState(
    bern: Bernoulli,
    measurement: Vector,
    H: Matrix,
    R: Matrix
  ): Bernoulli {
    return {
      r: 1,
      state: FinancialGaussianDensity.update(bern.state, measurement, H, R),
      t_birth: bern.t_birth,
      t_death: [bern.t_death[bern.t_death.length - 1]],
      w_death: [1]
    };
  }

  private pppDetectedUpdate(
    k: number,
    gatingRow: boolean[],
    measurement: Vector,
    H: Matrix,
    R: Matrix,
    P_D: number,
    clutterIntensity: number
  ): { bern: Bernoulli | null; logLik: number } {
    const gatedIndices = gatingRow
      .map((isInGate, idx) => (isInGate ? idx : -1))
      .filter(idx => idx >= 0);

    if (gatedIndices.length === 0) {
      return { bern: null, logLik: Math.log(clutterIntensity) };
    }

    const updatedStates = gatedIndices.map(idx =>
      FinancialGaussianDensity.update(this.params.PPP.states[idx], measurement, H, R)
    );
    const updatedWeights = gatedIndices.map(idx =>
      this.params.PPP.w[idx] +
      FinancialGaussianDensity.predictedLikelihood(this.params.PPP.states[idx], measurement, H, R) +
      Math.log(P_D)
    );

    const [normalizedWeights, logNumerator] = normalizeLogWeights(updatedWeights);
    const logLik = logSumExp([logNumerator, Math.log(clutterIntensity)]);

    return {
      bern: {
        r: Math.exp(logNumerator - logLik),
        state: FinancialGaussianDensity.momentMatching(normalizedWeights, updatedStates),
        t_birth: k,
        t_death: [k],
        w_death: [1]
      },
      logLik
    };
  }

  initialize(birthModel: { w: number; x: Vector; P: Matrix }[]) {
    this.params.PPP.w = birthModel.map(component => Math.log(component.w));
    this.params.PPP.states = birthModel.map(component =>
      this.cloneState({ x: component.x, P: component.P })
    );
    this.params.MBM = { w: [], ht: [], tt: [] };
  }

  predict(
    P_S: number,
    F: Matrix,
    Q: Matrix,
    birthModel: { w: number; x: Vector; P: Matrix }[],
    r_min: number
  ) {
    this.params.PPP.w = this.params.PPP.w.map(weight => weight + Math.log(P_S));
    this.params.PPP.states = this.params.PPP.states.map(state =>
      FinancialGaussianDensity.predict(state, F, Q)
    );

    this.params.PPP.w.push(...birthModel.map(component => Math.log(component.w)));
    this.params.PPP.states.push(
      ...birthModel.map(component => this.cloneState({ x: component.x, P: component.P }))
    );

    this.params.MBM.tt.forEach(tree => {
      tree.forEach(bern => {
        const aliveWeight = bern.w_death[bern.w_death.length - 1];
        if (aliveWeight >= r_min) {
          bern.state = FinancialGaussianDensity.predict(bern.state, F, Q);
          bern.t_death = [...bern.t_death, bern.t_death[bern.t_death.length - 1] + 1];
          bern.w_death = [
            ...bern.w_death.slice(0, -1),
            aliveWeight * (1 - P_S),
            aliveWeight * P_S
          ];
        }
      });
    });
  }

  update(
    k: number,
    measurements: Vector[],
    H: Matrix,
    R: Matrix,
    P_D: number,
    clutterIntensity: number,
    gatingSize: number,
    w_min: number,
    M: number
  ) {
    const measurementCount = measurements.length;
    const pppCount = this.params.PPP.states.length;
    const usedByPPP = Array(measurementCount).fill(false);
    const gatingPPP = Array.from({ length: measurementCount }, () => Array(pppCount).fill(false));

    for (let pppIdx = 0; pppIdx < pppCount; pppIdx++) {
      const inGate = FinancialGaussianDensity.ellipsoidalGating(
        this.params.PPP.states[pppIdx],
        measurements,
        H,
        R,
        gatingSize
      );
      inGate.forEach((isInGate, measIdx) => {
        gatingPPP[measIdx][pppIdx] = isInGate;
        usedByPPP[measIdx] = usedByPPP[measIdx] || isInGate;
      });
    }

    const treeCount = this.params.MBM.tt.length;
    const usedByDetectedTracks = Array(measurementCount).fill(false);
    const gatingDetected: boolean[][][] = Array.from({ length: treeCount }, () => []);

    for (let treeIdx = 0; treeIdx < treeCount; treeIdx++) {
      const localCount = this.params.MBM.tt[treeIdx].length;
      gatingDetected[treeIdx] = Array.from({ length: measurementCount }, () => Array(localCount).fill(false));
      for (let localIdx = 0; localIdx < localCount; localIdx++) {
        const inGate = FinancialGaussianDensity.ellipsoidalGating(
          this.params.MBM.tt[treeIdx][localIdx].state,
          measurements,
          H,
          R,
          gatingSize
        );
        inGate.forEach((isInGate, measIdx) => {
          gatingDetected[treeIdx][measIdx][localIdx] = isInGate;
          usedByDetectedTracks[measIdx] = usedByDetectedTracks[measIdx] || isInGate;
        });
      }
    }

    const usedOnlyByPPP = usedByPPP.map((used, measIdx) => used && !usedByDetectedTracks[measIdx]);
    const detectedMeasurementIndices = measurements
      .map((_, measIdx) => measIdx)
      .filter(measIdx => usedByDetectedTracks[measIdx]);
    const detectedMeasurements = detectedMeasurementIndices.map(measIdx => measurements[measIdx]);
    const gatedMeasurementCount = detectedMeasurements.length;

    const gatingDetectedUsed = gatingDetected.map(treeMatrix =>
      detectedMeasurementIndices.map(measIdx => treeMatrix[measIdx] ?? [])
    );
    const gatingPPPDetected = detectedMeasurementIndices.map(measIdx => gatingPPP[measIdx]);

    const updatedTreeCount = treeCount + gatedMeasurementCount;
    const hypothesisTable: (Bernoulli | null)[][] = Array.from({ length: updatedTreeCount }, () => []);
    const likelihoodTable: number[][][] = Array.from({ length: treeCount }, () => []);

    for (let treeIdx = 0; treeIdx < treeCount; treeIdx++) {
      const localCount = this.params.MBM.tt[treeIdx].length;
      likelihoodTable[treeIdx] = Array.from({ length: localCount }, () => Array(gatedMeasurementCount + 1).fill(-Infinity));
      hypothesisTable[treeIdx] = Array(localCount * (gatedMeasurementCount + 1)).fill(null);

      for (let localIdx = 0; localIdx < localCount; localIdx++) {
        const bern = this.params.MBM.tt[treeIdx][localIdx];
        const missed = this.bernoulliUndetectedUpdate(bern, P_D);
        hypothesisTable[treeIdx][localIdx * (gatedMeasurementCount + 1)] = missed.bern;
        likelihoodTable[treeIdx][localIdx][0] = missed.logLik;

        for (let measIdx = 0; measIdx < gatedMeasurementCount; measIdx++) {
          if (gatingDetectedUsed[treeIdx][measIdx]?.[localIdx]) {
            likelihoodTable[treeIdx][localIdx][measIdx + 1] = this.bernoulliDetectedLikelihood(
              bern,
              detectedMeasurements[measIdx],
              H,
              R,
              P_D
            );
            hypothesisTable[treeIdx][localIdx * (gatedMeasurementCount + 1) + measIdx + 1] =
              this.bernoulliDetectedUpdateState(bern, detectedMeasurements[measIdx], H, R);
          }
        }
      }
    }

    const newTrackLikelihoods = Array(gatedMeasurementCount).fill(Math.log(clutterIntensity));
    const detectedFromPPP = Array(gatedMeasurementCount).fill(false);

    for (let measIdx = 0; measIdx < gatedMeasurementCount; measIdx++) {
      const created = this.pppDetectedUpdate(
        k,
        gatingPPPDetected[measIdx] ?? [],
        detectedMeasurements[measIdx],
        H,
        R,
        P_D,
        clutterIntensity
      );
      if (created.bern) {
        hypothesisTable[treeCount + measIdx] = [created.bern];
        newTrackLikelihoods[measIdx] = created.logLik;
        detectedFromPPP[measIdx] = true;
      }
    }

    const clutterCost = Array.from({ length: gatedMeasurementCount }, (_, rowIdx) =>
      Array.from(
        { length: gatedMeasurementCount },
        (_, colIdx) => (rowIdx === colIdx ? -newTrackLikelihoods[rowIdx] : Infinity)
      )
    );

    let updatedGlobalWeights: number[] = [];
    let updatedGlobalTable: number[][] = [];
    const globalHypothesisCount = this.params.MBM.w.length;

    if (globalHypothesisCount === 0) {
      const initialTable = Array(updatedTreeCount).fill(0);
      for (let measIdx = 0; measIdx < gatedMeasurementCount; measIdx++) {
        if (detectedFromPPP[measIdx]) {
          initialTable[treeCount + measIdx] = 1;
        }
      }
      updatedGlobalWeights = [0];
      updatedGlobalTable = [initialTable];
    } else {
      for (let globalIdx = 0; globalIdx < globalHypothesisCount; globalIdx++) {
        const currentTable = this.params.MBM.ht[globalIdx] ?? [];
        const existingTrackCost = Array.from({ length: gatedMeasurementCount }, () => Array(treeCount).fill(Infinity));
        let missedDetectionWeight = 0;

        for (let treeIdx = 0; treeIdx < treeCount; treeIdx++) {
          const hypothesisIdx = currentTable[treeIdx] ?? 0;
          if (hypothesisIdx !== 0) {
            const localIdx = hypothesisIdx - 1;
            const missedLik = likelihoodTable[treeIdx][localIdx][0];
            missedDetectionWeight += missedLik;
            for (let measIdx = 0; measIdx < gatedMeasurementCount; measIdx++) {
              const detectedLik = likelihoodTable[treeIdx][localIdx][measIdx + 1];
              existingTrackCost[measIdx][treeIdx] = Number.isFinite(detectedLik)
                ? -(detectedLik - missedLik)
                : Infinity;
            }
          }
        }

        if (gatedMeasurementCount === 0) {
          const nextTable = currentTable.map(hypothesisIdx =>
            hypothesisIdx > 0 ? (hypothesisIdx - 1) * (gatedMeasurementCount + 1) + 1 : 0
          );
          updatedGlobalWeights.push(missedDetectionWeight + this.params.MBM.w[globalIdx]);
          updatedGlobalTable.push(nextTable);
          continue;
        }

        const costMatrix = existingTrackCost.map((row, rowIdx) => [...row, ...clutterCost[rowIdx]]);
        const assignmentLimit = Math.max(1, Math.ceil(Math.exp(this.params.MBM.w[globalIdx]) * M));
        const assignments = kBestAssignments(costMatrix, assignmentLimit);

        assignments.forEach(solution => {
          const assignedCols = new Map<number, number>();
          solution.assignment.forEach((colIdx, rowIdx) => {
            if (colIdx >= 0) {
              assignedCols.set(colIdx, rowIdx);
            }
          });

          const nextTable = Array(updatedTreeCount).fill(0);
          for (let treeIdx = 0; treeIdx < treeCount; treeIdx++) {
            const hypothesisIdx = currentTable[treeIdx] ?? 0;
            if (hypothesisIdx !== 0) {
              const assignedMeasurement = assignedCols.get(treeIdx);
              nextTable[treeIdx] =
                assignedMeasurement === undefined
                  ? (hypothesisIdx - 1) * (gatedMeasurementCount + 1) + 1
                  : (hypothesisIdx - 1) * (gatedMeasurementCount + 1) + assignedMeasurement + 2;
            }
          }
          for (let measIdx = 0; measIdx < gatedMeasurementCount; measIdx++) {
            const assignedMeasurement = assignedCols.get(treeCount + measIdx);
            if (assignedMeasurement !== undefined && detectedFromPPP[assignedMeasurement]) {
              nextTable[treeCount + measIdx] = 1;
            }
          }

          updatedGlobalWeights.push(-solution.cost + missedDetectionWeight + this.params.MBM.w[globalIdx]);
          updatedGlobalTable.push(nextTable);
        });
      }
    }

    const extraMeasurementIndices = measurements
      .map((_, measIdx) => measIdx)
      .filter(measIdx => usedOnlyByPPP[measIdx]);

    extraMeasurementIndices.forEach(measIdx => {
      const created = this.pppDetectedUpdate(
        k,
        gatingPPP[measIdx],
        measurements[measIdx],
        H,
        R,
        P_D,
        clutterIntensity
      );
      hypothesisTable.push(created.bern ? [created.bern] : []);
    });

    if (extraMeasurementIndices.length > 0) {
      updatedGlobalTable = updatedGlobalTable.map(row => [
        ...row,
        ...Array(extraMeasurementIndices.length).fill(1)
      ]);
    }

    this.params.PPP.w = this.params.PPP.w.map(weight => weight + Math.log(1 - P_D));

    const finiteGlobalHypotheses = updatedGlobalWeights
      .map((weight, idx) => ({ weight, idx }))
      .filter(({ weight }) => Number.isFinite(weight));

    if (finiteGlobalHypotheses.length === 0) {
      this.params.MBM = { w: [], ht: [], tt: [] };
      return;
    }

    const [normalizedFiniteWeights] = normalizeLogWeights(
      finiteGlobalHypotheses.map(({ weight }) => weight)
    );
    let normalizedGlobalHypotheses = finiteGlobalHypotheses.map(({ idx }, hypothesisIdx) => ({
      idx,
      weight: normalizedFiniteWeights[hypothesisIdx]
    }));

    normalizedGlobalHypotheses = normalizedGlobalHypotheses.filter(({ weight }) => weight >= w_min);

    if (normalizedGlobalHypotheses.length === 0) {
      const bestFiniteHypothesis = finiteGlobalHypotheses.reduce((best, candidate) =>
        candidate.weight > best.weight ? candidate : best
      );
      normalizedGlobalHypotheses = [{ idx: bestFiniteHypothesis.idx, weight: 0 }];
    }

    updatedGlobalWeights = normalizedGlobalHypotheses.map(({ weight }) => weight);
    updatedGlobalTable = normalizedGlobalHypotheses.map(({ idx }) => [...updatedGlobalTable[idx]]);

    if (updatedGlobalWeights.length > M) {
      const capped = updatedGlobalWeights
        .map((weight, idx) => ({ weight, idx }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, M);
      updatedGlobalWeights = capped.map(({ weight }) => weight);
      updatedGlobalTable = capped.map(({ idx }) => [...updatedGlobalTable[idx]]);
    }
    [updatedGlobalWeights] = normalizeLogWeights(updatedGlobalWeights);

    const totalTreeCount = updatedTreeCount + extraMeasurementIndices.length;
    const keepTreeMask = Array(totalTreeCount).fill(false);
    updatedGlobalTable.forEach(row => {
      for (let treeIdx = 0; treeIdx < totalTreeCount; treeIdx++) {
        keepTreeMask[treeIdx] = keepTreeMask[treeIdx] || (row[treeIdx] ?? 0) > 0;
      }
    });

    const filteredHypothesisTable = hypothesisTable.filter((_, treeIdx) => keepTreeMask[treeIdx]);
    const filteredGlobalTable = updatedGlobalTable.map(row => row.filter((_, treeIdx) => keepTreeMask[treeIdx]));
    const newTrees = filteredHypothesisTable.map((treeHypotheses, treeIdx) => {
      const referenced = stableUnique(
        filteredGlobalTable.map(row => row[treeIdx]).filter(localIdx => localIdx > 0)
      );
      return referenced
        .map(localIdx => treeHypotheses[localIdx - 1])
        .filter((bern): bern is Bernoulli => bern !== null)
        .map(bern => this.cloneBernoulli(bern));
    });

    const finalTreeCount = filteredGlobalTable.length > 0 ? filteredGlobalTable[0].length : 0;
    for (let treeIdx = 0; treeIdx < finalTreeCount; treeIdx++) {
      const activeIndices = stableUnique(
        filteredGlobalTable.map(row => row[treeIdx]).filter(localIdx => localIdx > 0)
      );
      const indexMap = new Map(activeIndices.map((localIdx, idx) => [localIdx, idx + 1]));
      filteredGlobalTable.forEach(row => {
        if ((row[treeIdx] ?? 0) > 0) {
          row[treeIdx] = indexMap.get(row[treeIdx]) ?? 0;
        }
      });
    }

    this.params.MBM.w = updatedGlobalWeights;
    this.params.MBM.ht = filteredGlobalTable;
    this.params.MBM.tt = newTrees;
  }

  prune(r_min: number, w_min: number) {
    for (let treeIdx = 0; treeIdx < this.params.MBM.tt.length; treeIdx++) {
      const tree = this.params.MBM.tt[treeIdx];
      const prunedLocalIndices: number[] = [];
      const keptBernoullis: Bernoulli[] = [];

      tree.forEach((bern, localIdx) => {
        if (bern.r < r_min) {
          prunedLocalIndices.push(localIdx + 1);
        } else {
          keptBernoullis.push(bern);
        }
      });

      this.params.MBM.tt[treeIdx] = keptBernoullis;
      prunedLocalIndices.forEach(localIdx => {
        this.params.MBM.ht.forEach(row => {
          if (row[treeIdx] === localIdx) {
            row[treeIdx] = 0;
          }
        });
      });
    }

    if (this.params.MBM.ht.length > 0) {
      const keepTreeMask = this.params.MBM.ht[0].map((_, treeIdx) =>
        this.params.MBM.ht.some(row => row[treeIdx] !== 0)
      );
      this.params.MBM.ht = this.params.MBM.ht.map(row => row.filter((_, treeIdx) => keepTreeMask[treeIdx]));
      this.params.MBM.tt = this.params.MBM.tt.filter((_, treeIdx) => keepTreeMask[treeIdx]);
    }

    if (
      this.params.MBM.ht.length === 0 ||
      (this.params.MBM.ht[0] !== undefined && this.params.MBM.ht[0].length === 0)
    ) {
      this.params.MBM = { w: [], ht: [], tt: [] };
    } else {
      for (let treeIdx = 0; treeIdx < this.params.MBM.tt.length; treeIdx++) {
        const activeIndices = stableUnique(
          this.params.MBM.ht.map(row => row[treeIdx]).filter(localIdx => localIdx > 0)
        );
        const indexMap = new Map(activeIndices.map((localIdx, idx) => [localIdx, idx + 1]));
        this.params.MBM.ht.forEach(row => {
          if (row[treeIdx] > 0) {
            row[treeIdx] = indexMap.get(row[treeIdx]) ?? 0;
          }
        });
      }

      const uniqueRows: number[][] = [];
      const uniqueWeights: number[] = [];
      const rowGroups = new Map<string, number[]>();

      this.params.MBM.ht.forEach((row, rowIdx) => {
        const key = row.join(",");
        if (!rowGroups.has(key)) {
          rowGroups.set(key, []);
          uniqueRows.push([...row]);
        }
        rowGroups.get(key)?.push(rowIdx);
      });

      uniqueRows.forEach(row => {
        const duplicateIndices = rowGroups.get(row.join(",")) ?? [];
        uniqueWeights.push(logSumExp(duplicateIndices.map(idx => this.params.MBM.w[idx])));
      });

      this.params.MBM.ht = uniqueRows;
      this.params.MBM.w = uniqueWeights;
    }

    const validPPP = this.params.PPP.w
      .map((weight, idx) => ({ weight, state: this.params.PPP.states[idx] }))
      .filter(({ weight }) => weight >= w_min);

    this.params.PPP.w = validPPP.map(({ weight }) => weight);
    this.params.PPP.states = validPPP.map(({ state }) => this.cloneState(state));
  }

  extractBestEstimate(minExistence = 0.2): FinancialEstimate | null {
    if (this.params.MBM.w.length === 0 || this.params.MBM.ht.length === 0) {
      return null;
    }

    const bestGlobalIdx = this.params.MBM.w.indexOf(Math.max(...this.params.MBM.w));
    const bestHypothesis = this.params.MBM.ht[bestGlobalIdx] ?? [];

    let bestBernoulli: Bernoulli | null = null;
    let bestScore = -Infinity;

    for (let treeIdx = 0; treeIdx < bestHypothesis.length && treeIdx < this.params.MBM.tt.length; treeIdx++) {
      const localIdx = bestHypothesis[treeIdx];
      if (localIdx > 0 && localIdx <= this.params.MBM.tt[treeIdx].length) {
        const bern = this.params.MBM.tt[treeIdx][localIdx - 1];
        const score = bern.r * bern.w_death[bern.w_death.length - 1];
        if (score > bestScore) {
          bestScore = score;
          bestBernoulli = bern;
        }
      }
    }

    if (!bestBernoulli || bestBernoulli.r < minExistence) {
      return null;
    }

    return {
      price: bestBernoulli.state.x[0],
      velocity: bestBernoulli.state.x[1],
      existence: bestBernoulli.r,
      variance: bestBernoulli.state.P[0][0]
    };
  }
}

interface Signal {
  time: number;
  price: number;
  type: 'buy' | 'sell';
  shares?: number;
  cashAfter?: number;
  sharesAfter?: number;
  reason?: string;
}

interface Portfolio {
  cash: number;
  shares: number;
  totalValue: number;
}

const FinancialTracker: React.FC = () => {
  const [noiseLevel, setNoiseLevel] = useState<'low' | 'medium' | 'high'>('medium');
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(true);
  const [instrument, setInstrument] = useState<'stock' | 'crypto' | 'forex'>('stock');
  const [startingCapital, setStartingCapital] = useState(10000);

  const results = useMemo(() => {
    const N = 500;
    const dt = 1;
    const priceMultiplier = instrument === 'stock' ? 1 : instrument === 'crypto' ? 500 : 0.01;
    const startingPrice = instrument === 'stock' ? 100 : instrument === 'crypto' ? 50000 : 1.2;
    const minimumPrice = instrument === 'stock' ? 5 : instrument === 'crypto' ? 5000 : 0.4;

    const seed = hashString(`${instrument}-${noiseLevel}`);
    const rng = createSeededRandom(seed);
    const normal = () => seededRandomNormal(rng);

    let price = startingPrice;
    let velocity = 0;
    const times = Array.from({ length: N }, (_, i) => i);
    const truePrices: number[] = [];

    for (let i = 0; i < N; i++) {
      if (i === 100) velocity += 0.3 * priceMultiplier;
      if (i === 200) velocity -= 0.5 * priceMultiplier;
      if (i === 300) velocity += 0.4 * priceMultiplier;
      if (i === 400) velocity -= 0.3 * priceMultiplier;

      velocity *= 0.99;
      velocity += normal() * 0.05 * priceMultiplier;
      price = Math.max(minimumPrice, price + velocity);
      truePrices.push(price);
    }

    const noiseConfig = {
      low: { sigma: 0.3, outlierProbability: 0.02, outlierScale: 2.5 },
      medium: { sigma: 0.8, outlierProbability: 0.05, outlierScale: 4.0 },
      high: { sigma: 1.5, outlierProbability: 0.09, outlierScale: 6.0 }
    };
    const processNoiseConfig = {
      low: { qPos: 0.01, qVel: 0.001 },
      medium: { qPos: 0.1, qVel: 0.01 },
      high: { qPos: 0.5, qVel: 0.05 }
    };

    const measurementSettings = noiseConfig[noiseLevel];
    const measurementNoise = measurementSettings.sigma * priceMultiplier;
    const observations = truePrices.map((truePrice, idx) => {
      let observedPrice = truePrice + normal() * measurementNoise;
      if (rng() < measurementSettings.outlierProbability) {
        const trendDirection = idx > 0 ? Math.sign(truePrice - truePrices[idx - 1]) || 1 : 1;
        observedPrice +=
          trendDirection *
          measurementSettings.outlierScale *
          measurementNoise *
          (0.5 + Math.abs(normal()));
      }
      return Math.max(minimumPrice, observedPrice);
    });

    const { qPos, qVel } = processNoiseConfig[noiseLevel];
    const F = [
      [1, dt],
      [0, 0.98]
    ];
    const H = [[1, 0]];
    const Q = [
      [qPos * priceMultiplier * priceMultiplier, 0],
      [0, qVel * priceMultiplier * priceMultiplier]
    ];
    const R = [[measurementNoise * measurementNoise]];

    const P_S = 0.995;
    const P_D = noiseLevel === 'high' ? 0.94 : noiseLevel === 'medium' ? 0.97 : 0.99;
    const gatingSize = 10.827566170662733;
    const w_min = Math.log(1e-4);
    const r_min = 1e-3;
    const M = 20;
    const clutterLambda = noiseLevel === 'high' ? 0.45 : noiseLevel === 'medium' ? 0.25 : 0.1;
    const measurementDomainMin = Math.min(...truePrices, ...observations) - 8 * measurementNoise;
    const measurementDomainMax = Math.max(...truePrices, ...observations) + 8 * measurementNoise;
    const clutterIntensity = clutterLambda / Math.max(measurementDomainMax - measurementDomainMin, 1e-6);

    const makeBirthModel = (anchorPrice: number, anchorVelocity: number) => [
      {
        w: 0.04,
        x: [anchorPrice, anchorVelocity * 0.5],
        P: [
          [Math.max(measurementNoise * measurementNoise * 9, 1e-4), 0],
          [0, Math.max(Q[1][1] * 50, priceMultiplier * priceMultiplier * 0.02)]
        ]
      },
      {
        w: 0.02,
        x: [anchorPrice, 0],
        P: [
          [Math.max(measurementNoise * measurementNoise * 16, 1e-4), 0],
          [0, Math.max(Q[1][1] * 100, priceMultiplier * priceMultiplier * 0.05)]
        ]
      }
    ];

    const filter = new FinancialPMBMFilter();
    filter.initialize(makeBirthModel(observations[0], 0));

    const filteredPrices: number[] = [];
    const filteredVelocities: number[] = [];
    const filteredConfidence: number[] = [];
    const filteredVariance: number[] = [];

    let fallbackPrice = observations[0];
    let fallbackVelocity = 0;

    for (let i = 0; i < N; i++) {
      filter.update(
        i + 1,
        [[observations[i]]],
        H,
        R,
        P_D,
        clutterIntensity,
        gatingSize,
        w_min,
        M
      );

      const estimate = filter.extractBestEstimate(0.15);
      const currentEstimate = estimate ?? {
        price: fallbackPrice,
        velocity: fallbackVelocity,
        existence: 0,
        variance: filteredVariance[i - 1] ?? R[0][0] * 4
      };

      fallbackPrice = currentEstimate.price;
      fallbackVelocity = currentEstimate.velocity;
      filteredPrices.push(currentEstimate.price);
      filteredVelocities.push(currentEstimate.velocity);
      filteredConfidence.push(currentEstimate.existence);
      filteredVariance.push(currentEstimate.variance);

      filter.prune(r_min, w_min);
      filter.predict(
        P_S,
        F,
        Q,
        makeBirthModel(currentEstimate.price, currentEstimate.velocity),
        r_min
      );
    }

    const averageWindow = (series: number[], endIdx: number, window: number): number => {
      const startIdx = Math.max(0, endIdx - window + 1);
      let sum = 0;
      for (let idx = startIdx; idx <= endIdx; idx++) {
        sum += series[idx];
      }
      return sum / (endIdx - startIdx + 1);
    };

    const averageResidual = (endIdx: number, window: number): number => {
      const startIdx = Math.max(0, endIdx - window + 1);
      let sum = 0;
      for (let idx = startIdx; idx <= endIdx; idx++) {
        sum += Math.abs(observations[idx] - filteredPrices[idx]);
      }
      return sum / (endIdx - startIdx + 1);
    };

    const signals: Signal[] = [];
    const portfolioHistory: Portfolio[] = [];
    const positionStep = instrument === 'stock' ? 1 : instrument === 'crypto' ? 0.001 : 1000;
    const tradeBudgetFraction = instrument === 'crypto' ? 0.92 : 0.95;
    const baseVelocityThreshold = 0.04 * priceMultiplier;
    const minimumConfidence = noiseLevel === 'high' ? 0.52 : 0.58;
    const stopLossPct = instrument === 'crypto' ? 0.08 : instrument === 'forex' ? 0.02 : 0.05;
    const trailingStopPct = instrument === 'crypto' ? 0.12 : instrument === 'forex' ? 0.015 : 0.07;
    const cooldownPeriod = 18;
    const minimumHoldPeriod = 10;

    let cash = startingCapital;
    let shares = 0;
    let lastTradeTime = -Infinity;
    let entryTime: number | null = null;
    let entryPrice: number | null = null;
    let peakSinceEntry = -Infinity;

    for (let i = 0; i < N; i++) {
      const currentPrice = filteredPrices[i];
      const executionPrice = truePrices[i];
      const currentConfidence = filteredConfidence[i];
      const currentVariance = filteredVariance[i];
      const trendScore = averageWindow(filteredVelocities, i, 5);
      const previousTrendScore = i > 0 ? averageWindow(filteredVelocities, i - 1, 5) : 0;
      const shortAverage = averageWindow(filteredPrices, i, 6);
      const longAverage = averageWindow(filteredPrices, i, 18);
      const adaptiveThreshold = Math.max(baseVelocityThreshold, averageResidual(i, 8) * 0.18);
      const confidenceIsUsable = currentConfidence >= minimumConfidence && currentVariance < R[0][0] * 8;
      const bullishCross =
        i >= 12 &&
        previousTrendScore <= adaptiveThreshold &&
        trendScore > adaptiveThreshold &&
        currentPrice > shortAverage &&
        shortAverage >= longAverage;
      const bearishCross =
        i >= 12 &&
        previousTrendScore >= -adaptiveThreshold &&
        trendScore < -adaptiveThreshold &&
        currentPrice < shortAverage &&
        shortAverage <= longAverage;

      if (shares > 0) {
        peakSinceEntry = Math.max(peakSinceEntry, executionPrice);
      }

      const canEnter = i - lastTradeTime >= cooldownPeriod;
      const heldLongEnough = entryTime === null || i - entryTime >= minimumHoldPeriod;
      const stopLossTriggered =
        shares > 0 && entryPrice !== null && executionPrice <= entryPrice * (1 - stopLossPct);
      const trailingStopTriggered =
        shares > 0 &&
        Number.isFinite(peakSinceEntry) &&
        executionPrice <= peakSinceEntry * (1 - trailingStopPct);
      const confidenceBreakdown = shares > 0 && currentConfidence < 0.35;
      const riskExitTriggered = stopLossTriggered || trailingStopTriggered || confidenceBreakdown;
      const reversalExitTriggered = heldLongEnough && bearishCross;

      if (shares === 0 && canEnter && confidenceIsUsable && bullishCross) {
        const budget = cash * tradeBudgetFraction;
        const sharesToBuy = roundDownToStep(budget / executionPrice, positionStep);
        if (sharesToBuy >= positionStep) {
          const cost = sharesToBuy * executionPrice;
          cash -= cost;
          shares += sharesToBuy;
          lastTradeTime = i;
          entryTime = i;
          entryPrice = executionPrice;
          peakSinceEntry = executionPrice;
          signals.push({
            time: i,
            price: executionPrice,
            type: 'buy',
            shares: sharesToBuy,
            cashAfter: cash,
            sharesAfter: shares,
            reason: 'Trend breakout confirmed by PMBM confidence'
          });
        }
      } else if (shares > 0 && (riskExitTriggered || reversalExitTriggered)) {
        const proceeds = shares * executionPrice;
        const sharesSold = shares;
        cash += proceeds;
        shares = 0;
        lastTradeTime = i;
        entryTime = null;
        entryPrice = null;
        peakSinceEntry = -Infinity;
        signals.push({
          time: i,
          price: executionPrice,
          type: 'sell',
          shares: sharesSold,
          cashAfter: cash,
          sharesAfter: shares,
          reason: stopLossTriggered
            ? 'Protective stop loss'
            : trailingStopTriggered
              ? 'Trailing stop locked in gains'
              : confidenceBreakdown
                ? 'Track confidence breakdown'
                : 'Trend reversal confirmed'
        });
      }

      portfolioHistory.push({
        cash,
        shares,
        totalValue: cash + shares * executionPrice
      });
    }

    return {
      times,
      truePrices,
      observations,
      filteredPrices,
      filteredConfidence,
      signals,
      portfolioHistory,
      startingCapital
    };
  }, [noiseLevel, instrument, startingCapital]);

  const maxStep = results.times.length - 1;

  useEffect(() => {
    if (isAnimating && animationStep < maxStep) {
      const timer = setTimeout(() => {
        setAnimationStep(prev => {
          const next = Math.min(prev + 2, maxStep);
          if (next >= maxStep) {
            setIsAnimating(false);
          }
          return next;
        });
      }, 20);
      return () => clearTimeout(timer);
    }
  }, [isAnimating, animationStep, maxStep]);

  const handleAnimate = () => {
    setAnimationStep(0);
    setIsAnimating(true);
  };

  const handleReset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
  };

  const displayStep = isAnimating ? animationStep : maxStep;
  const currentPortfolio = results.portfolioHistory[Math.min(displayStep, results.portfolioHistory.length - 1)];
  const profitLoss = currentPortfolio ? currentPortfolio.totalValue - results.startingCapital : 0;
  const profitLossPercent = currentPortfolio ? ((currentPortfolio.totalValue - results.startingCapital) / results.startingCapital * 100) : 0;
  const currentConfidence = results.filteredConfidence[Math.min(displayStep, results.filteredConfidence.length - 1)] ?? 0;
  const priceDigits = instrument === 'forex' ? 4 : instrument === 'crypto' ? 0 : 2;
  const quantityDigits = instrument === 'stock' ? 0 : instrument === 'crypto' ? 3 : 0;
  const quantityLabel = instrument === 'forex' ? 'Units' : instrument === 'crypto' ? 'Coins' : 'Shares';
  
  const minPrice = Math.min(...results.truePrices, ...results.observations, ...results.filteredPrices);
  const maxPrice = Math.max(...results.truePrices, ...results.observations, ...results.filteredPrices);
  const priceRange = maxPrice - minPrice;
  const padding = priceRange * 0.1;
  
  const viewBox = {
    minX: -20,
    minY: -(maxPrice + padding),
    width: results.times.length + 40,
    height: priceRange + 2 * padding
  };

  const visibleSignals = results.signals.filter(s => s.time <= displayStep);

  return (
    <main className="page-shell demo-shell">
      <DemoShellStyles />

      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Financial Tracking / PMBM</p>
          <h1>Financial PMBM Tracker</h1>
          <p className="lede">
            Poisson multi-Bernoulli mixture filtering separates persistent price motion from noisy
            ticks before the strategy evaluates entries, exits, and portfolio value.
          </p>
        </div>

        <div className="control-panel">
          <div>
            <p className="mini-label">Instrument</p>
            <div className="button-row mt-3">
              <button
                type="button"
                onClick={() => setInstrument('stock')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  instrument === 'stock'
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Stock
              </button>
            </div>
          </div>

          <div>
            <p className="mini-label">Noise Profile</p>
            <div className="button-row mt-3">
              <button
                type="button"
                onClick={() => setNoiseLevel('low')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  noiseLevel === 'low'
                    ? 'bg-emerald-600 text-white shadow-md'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Low Noise
              </button>
              <button
                type="button"
                onClick={() => setNoiseLevel('medium')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  noiseLevel === 'medium'
                    ? 'bg-emerald-600 text-white shadow-md'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Medium Noise
              </button>
              <button
                type="button"
                onClick={() => setNoiseLevel('high')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  noiseLevel === 'high'
                    ? 'bg-emerald-600 text-white shadow-md'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                High Noise
              </button>
            </div>
          </div>

          <div className="button-row items-center">
            <label className="text-slate-300 text-sm font-medium">Capital:</label>
            <input
              type="number"
              value={startingCapital}
              onChange={(e) => setStartingCapital(Number(e.target.value))}
              className="w-32 px-3 py-2 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
              min="100"
              step="1000"
            />
            <button
              type="button"
              onClick={handleAnimate}
              disabled={isAnimating}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-slate-600 transition-all shadow-md"
            >
              {isAnimating ? 'Animating...' : 'Animate'}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-6 py-2 bg-slate-600 text-white rounded-lg font-medium hover:bg-slate-500 transition-all shadow-md"
            >
              Reset
            </button>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <article className="canvas-card">
          <div className="section-head">
            <div className="flex items-center gap-3">
              <DollarSign className="w-7 h-7 text-emerald-400" />
              <div>
                <p className="mini-label">Price Simulation</p>
                <h2>Observed market vs PMBM estimate</h2>
              </div>
            </div>
            <p className="annotation">
              The chart keeps the filtered estimate, observations, and generated trading signals on
              one dark surface for quick comparison.
            </p>
          </div>

          {currentPortfolio && (
            <div className="surface-inset mb-4 grid grid-cols-2 gap-4 p-4 md:grid-cols-5">
              <div>
                <div className="text-xs text-slate-400 mb-1">Cash</div>
                <div className="text-lg font-bold text-white">${currentPortfolio.cash.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">{quantityLabel}</div>
                <div className="text-lg font-bold text-white">{currentPortfolio.shares.toFixed(quantityDigits)}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">Total Value</div>
                <div className="text-lg font-bold text-white">${currentPortfolio.totalValue.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">Track Confidence</div>
                <div className="text-lg font-bold text-white">{(currentConfidence * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">P/L</div>
                <div className={`text-lg font-bold ${profitLoss >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {profitLoss >= 0 ? '+' : ''}{profitLoss.toFixed(2)} ({profitLossPercent >= 0 ? '+' : ''}{profitLossPercent.toFixed(2)}%)
                </div>
              </div>
            </div>
          )}

          <div className="viz-surface p-3 sm:p-4">
            <svg
              viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`}
              className="w-full h-auto"
              style={{ aspectRatio: `${viewBox.width} / ${viewBox.height}` }}
            >
              <defs>
                <linearGradient id="priceGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
                </linearGradient>
              </defs>

              <rect
                x={viewBox.minX}
                y={viewBox.minY}
                width={viewBox.width}
                height={viewBox.height}
                fill="#07111d"
              />

              <g opacity="0.18">
                {Array.from({ length: 11 }, (_, i) => (
                  <line
                    key={`v-${i}`}
                    x1={i * 50}
                    y1={viewBox.minY}
                    x2={i * 50}
                    y2={-viewBox.minY}
                    stroke="#334155"
                    strokeWidth="0.5"
                  />
                ))}
                {Array.from({ length: 8 }, (_, i) => {
                  const y = -(minPrice + i * priceRange / 7);
                  return (
                    <line
                      key={`h-${i}`}
                      x1={viewBox.minX}
                      y1={y}
                      x2={results.times.length}
                      y2={y}
                      stroke="#334155"
                      strokeWidth="0.5"
                    />
                  );
                })}
              </g>

              <line
                x1={0}
                y1={viewBox.minY}
                x2={0}
                y2={-viewBox.minY}
                stroke="#94a3b8"
                strokeWidth="2"
              />

              {Array.from({ length: 8 }, (_, i) => {
                const price = minPrice + i * priceRange / 7;
                const y = -price;
                return (
                  <text
                    key={`price-${i}`}
                    x={-5}
                    y={y + 3}
                    fontSize="10"
                    fill="#cbd5e1"
                    textAnchor="end"
                  >
                    {price.toFixed(priceDigits)}
                  </text>
                );
              })}

              <path
                d={results.truePrices.slice(0, displayStep + 1).map((price, i) =>
                  `${i === 0 ? 'M' : 'L'} ${i} ${-price}`
                ).join(' ')}
                stroke="#64748b"
                strokeWidth="1"
                fill="none"
                opacity="0.3"
                strokeDasharray="2,2"
              />

              <path
                d={results.observations.slice(0, displayStep + 1).map((price, i) =>
                  `${i === 0 ? 'M' : 'L'} ${i} ${-price}`
                ).join(' ')}
                stroke="#60a5fa"
                strokeWidth="1.5"
                fill="none"
                opacity="0.7"
              />

              <path
                d={results.filteredPrices.slice(0, displayStep + 1).map((price, i) =>
                  `${i === 0 ? 'M' : 'L'} ${i} ${-price}`
                ).join(' ')}
                stroke="#10b981"
                strokeWidth="2.5"
                fill="none"
              />

              {visibleSignals.filter(s => s.type === 'buy').map((signal, idx) => (
                <g key={`buy-${idx}`}>
                  <circle
                    cx={signal.time}
                    cy={-signal.price}
                    r="4"
                    fill="#10b981"
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <TrendingUp
                    x={signal.time - 8}
                    y={-signal.price - 20}
                    width={16}
                    height={16}
                    stroke="#10b981"
                    strokeWidth={2.5}
                    fill="none"
                  />
                </g>
              ))}

              {visibleSignals.filter(s => s.type === 'sell').map((signal, idx) => (
                <g key={`sell-${idx}`}>
                  <circle
                    cx={signal.time}
                    cy={-signal.price}
                    r="4"
                    fill="#ef4444"
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <TrendingDown
                    x={signal.time - 8}
                    y={-signal.price + 8}
                    width={16}
                    height={16}
                    stroke="#ef4444"
                    strokeWidth={2.5}
                    fill="none"
                  />
                </g>
              ))}

              {displayStep > 0 && (
                <circle
                  cx={displayStep}
                  cy={-results.filteredPrices[displayStep]}
                  r="3"
                  fill="#fbbf24"
                  stroke="#fff"
                  strokeWidth="2"
                />
              )}
            </svg>
          </div>

          <div className="legend-strip mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-6 h-0.5 bg-slate-500 opacity-50" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #64748b 0, #64748b 3px, transparent 3px, transparent 6px)' }}></div>
              <span className="text-slate-300">True Price</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-0.5 bg-blue-400 opacity-60"></div>
              <span className="text-slate-300">Observed Price</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-emerald-500"></div>
              <span className="text-slate-300">PMBM Estimate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-emerald-500"></div>
              <TrendingUp className="w-4 h-4 text-emerald-500" />
              <span className="text-slate-300">Buy Signal</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <TrendingDown className="w-4 h-4 text-red-500" />
              <span className="text-slate-300">Sell Signal</span>
            </div>
          </div>
        </article>

        <aside className="sidebar">
          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Trading Signals</p>
                <h2>Latest actions</h2>
              </div>
            </div>
            <div className="space-y-3 max-h-60 overflow-y-auto">
              {visibleSignals.length === 0 && (
                <p className="text-slate-400">No signals generated yet...</p>
              )}
              {visibleSignals.slice().reverse().map((signal, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border-l-4 ${
                    signal.type === 'buy'
                      ? 'bg-emerald-900/30 border-emerald-500'
                      : 'bg-red-900/30 border-red-500'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {signal.type === 'buy' ? (
                        <TrendingUp className="w-5 h-5 text-emerald-400" />
                      ) : (
                        <TrendingDown className="w-5 h-5 text-red-400" />
                      )}
                      <span className="font-semibold text-white uppercase">
                        {signal.type}
                      </span>
                    </div>
                    <span className="text-slate-300">
                      ${signal.price.toFixed(priceDigits)}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {quantityLabel}: {signal.shares?.toFixed(quantityDigits)} | Time: {signal.time}
                  </div>
                  <div className="text-xs text-slate-300 mt-1">
                    Cash: ${signal.cashAfter?.toFixed(2)} | Holding: {signal.sharesAfter?.toFixed(quantityDigits)} {quantityLabel.toLowerCase()}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {signal.reason}
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">About This System</p>
                <h2>Model and execution</h2>
              </div>
            </div>
            <div className="text-slate-300 space-y-2 text-sm">
              <p>This system uses a <strong>PMBM filter</strong> to separate persistent price motion from noisy ticks and synthetic clutter before generating trading decisions.</p>
              <p><strong>State Model:</strong> Latent price and velocity are tracked with a linear-Gaussian motion model inside a multi-hypothesis PMBM update.</p>
              <p><strong>Buy Logic:</strong> Enters only after a filtered trend breakout, minimum confidence, and cooldown clearance.</p>
              <p><strong>Sell Logic:</strong> Trend reversals respect a hold period, while stop loss, trailing stop, and confidence breakdown exits can trigger immediately.</p>
              <p><strong>Execution Logic:</strong> Signals come from the PMBM estimate, but trades and portfolio valuation use the simulated market price rather than the filtered estimate.</p>
              <p><strong>Noise Levels:</strong> Simulate different market conditions and volatility.</p>
              <p className="text-yellow-400 mt-3">For educational purposes only. Not financial advice.</p>
            </div>
          </article>
        </aside>
      </section>
    </main>
  );
};

export default FinancialTracker;
