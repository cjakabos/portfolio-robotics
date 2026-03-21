'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, Settings, Eye, EyeOff, Info } from 'lucide-react';
import DemoShellStyles from '../_components/demo-shell-styles';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

type Matrix = number[][];
type Vector = number[];

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
    states: GaussianState[] 
  };
  MBM: { 
    w: number[];
    ht: number[][];
    tt: Bernoulli[][];  // Array of arrays of Bernoulli, not array of objects
  };
}

interface MotionModel {
  d: number;
  F: Matrix;
  Q: Matrix;
}

interface MeasModel {
  d: number;
  H: Matrix;
  R: Matrix;
}

interface GroundTruth {
  nbirths: number;
  xstart: Vector[];
  tbirth: number[];
  tdeath: number[];
  K: number;
}

interface ObjectData {
  X: Vector[][];
  N: number[];
}

// ============================================================================
// LINEAR ALGEBRA UTILITIES
// ============================================================================

const transpose = (m: Matrix): Matrix => 
  m[0].map((_, i) => m.map(row => row[i]));

const mult = (a: Matrix, b: Matrix): Matrix => {
  const result: Matrix = Array(a.length).fill(0).map(() => Array(b[0].length).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b[0].length; j++) {
      for (let k = 0; k < b.length; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
};

const add = (a: Matrix, b: Matrix): Matrix =>
  a.map((row, i) => row.map((val, j) => val + b[i][j]));

const subtract = (a: Matrix, b: Matrix): Matrix =>
  a.map((row, i) => row.map((val, j) => val - b[i][j]));

const scale = (s: number, m: Matrix): Matrix =>
  m.map(row => row.map(val => val * s));

const identity = (n: number): Matrix =>
  Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));

const inv2x2 = (m: Matrix): Matrix => {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (Math.abs(det) < 1e-10) return identity(2);
  return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
};

const normalizeLogWeights = (logWeights: number[]): [number[], number] => {
  if (logWeights.length === 0) return [[], -Infinity];
  if (logWeights.length === 1) return [[0], logWeights[0]];
  
  const maxLog = Math.max(...logWeights);
  const logSumW = maxLog + Math.log(logWeights.reduce((sum, w) => sum + Math.exp(w - maxLog), 0));
  return [logWeights.map(w => w - logSumW), logSumW];
};

const logSumExp = (logWeights: number[]): number =>
  normalizeLogWeights(logWeights)[1];

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

// ============================================================================
// M-BEST ASSIGNMENT SEARCH FOR PMBM COST MATRICES
// ============================================================================

interface AssignmentSolution {
  assignment: number[];
  cost: number;
}

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
  const m = costMatrix.length;
  if (m === 0) return [{ assignment: [], cost: 0 }];
  if (k <= 0) return [];
  
  const n = costMatrix[0].length;
  const nExisting = Math.max(0, n - m);
  const optionsByRow = Array.from({ length: m }, (_, rowIdx) => {
    const options: { col: number; cost: number }[] = [];
    for (let colIdx = 0; colIdx < nExisting; colIdx++) {
      const cost = costMatrix[rowIdx][colIdx];
      if (Number.isFinite(cost)) {
        options.push({ col: colIdx, cost });
      }
    }
    const fallbackCol = nExisting + rowIdx;
    if (fallbackCol < n && Number.isFinite(costMatrix[rowIdx][fallbackCol])) {
      options.push({ col: fallbackCol, cost: costMatrix[rowIdx][fallbackCol] });
    }
    options.sort((a, b) => a.cost - b.cost);
    return options;
  });
  
  const rowOrder = Array.from({ length: m }, (_, rowIdx) => rowIdx).sort((a, b) => {
    const optionCountDiff = optionsByRow[a].length - optionsByRow[b].length;
    if (optionCountDiff !== 0) return optionCountDiff;
    return optionsByRow[a][0].cost - optionsByRow[b][0].cost;
  });
  
  const orderedOptions = rowOrder.map(rowIdx => optionsByRow[rowIdx]);
  const usedExisting = Array(nExisting).fill(false);
  const assignment = Array(m).fill(-1);
  const solutions: AssignmentSolution[] = [];
  
  const lowerBound = (startIdx: number): number => {
    let bound = 0;
    for (let idx = startIdx; idx < orderedOptions.length; idx++) {
      const bestOption = orderedOptions[idx].find(
        option => option.col >= nExisting || !usedExisting[option.col]
      );
      if (!bestOption) return Infinity;
      bound += bestOption.cost;
    }
    return bound;
  };
  
  const search = (depth: number, currentCost: number) => {
    const optimisticCost = currentCost + lowerBound(depth);
    const worstAcceptedCost =
      solutions.length === k ? solutions[solutions.length - 1].cost : Infinity;
    
    if (!Number.isFinite(optimisticCost) || optimisticCost > worstAcceptedCost) {
      return;
    }
    
    if (depth === m) {
      insertAssignmentSolution(solutions, { assignment: [...assignment], cost: currentCost }, k);
      return;
    }
    
    const rowIdx = rowOrder[depth];
    for (const option of orderedOptions[depth]) {
      if (option.col < nExisting && usedExisting[option.col]) continue;
      
      if (option.col < nExisting) {
        usedExisting[option.col] = true;
      }
      assignment[rowIdx] = option.col;
      search(depth + 1, currentCost + option.cost);
      assignment[rowIdx] = -1;
      if (option.col < nExisting) {
        usedExisting[option.col] = false;
      }
    }
  };
  
  search(0, 0);
  return solutions;
};

const sampleStandardNormal = (() => {
  let spare: number | null = null;
  
  return () => {
    if (spare !== null) {
      const cached = spare;
      spare = null;
      return cached;
    }
    
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
})();

const samplePoisson = (lambda: number): number => {
  if (lambda <= 0) return 0;
  const threshold = Math.exp(-lambda);
  let product = 1;
  let count = 0;
  
  do {
    count++;
    product *= Math.random();
  } while (product > threshold);
  
  return count - 1;
};

// ============================================================================
// GAUSSIAN DENSITY CLASS
// ============================================================================

class GaussianDensity {
  static predict(state: GaussianState, F: Matrix, Q: Matrix): GaussianState {
    const x = mult(F, [[state.x[0]], [state.x[1]], [state.x[2]], [state.x[3]]]).map(r => r[0]);
    const P = add(mult(mult(F, state.P), transpose(F)), Q);
    return { x, P };
  }

  static update(state: GaussianState, z: Vector, H: Matrix, R: Matrix): GaussianState {
    const HxP = mult(H, state.P);
    const S = add(mult(HxP, transpose(H)), R);
    const SInv = inv2x2(S);
    const K = mult(mult(state.P, transpose(H)), SInv);
    
    const zPred = mult(H, [[state.x[0]], [state.x[1]], [state.x[2]], [state.x[3]]]).map(r => r[0]);
    const innovation = z.map((zi, i) => zi - zPred[i]);
    const x = state.x.map((xi, i) => xi + K[i].reduce((sum, k, j) => sum + k * innovation[j], 0));
    
    const IminusKH = subtract(identity(4), mult(K, H));
    const P = mult(IminusKH, state.P);
    
    return { x, P };
  }

  static predictedLikelihood(state: GaussianState, z: Vector, H: Matrix, R: Matrix): number {
    const zPred = mult(H, [[state.x[0]], [state.x[1]], [state.x[2]], [state.x[3]]]).map(r => r[0]);
    const HxP = mult(H, state.P);
    const S = add(mult(HxP, transpose(H)), R);
    
    const diff = z.map((zi, i) => zi - zPred[i]);
    const SInv = inv2x2(S);
    const mahalDist = diff.reduce((sum, di, i) => 
      sum + di * diff.reduce((s, dj, j) => s + SInv[i][j] * dj, 0), 0);
    
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (det <= 0) return -1000;
    
    return -0.5 * (mahalDist + Math.log(det) + 2 * Math.log(2 * Math.PI));
  }

  static ellipsoidalGating(state: GaussianState, z: Vector[], H: Matrix, R: Matrix, gatingSize: number): boolean[] {
    const HxP = mult(H, state.P);
    const S = add(mult(HxP, transpose(H)), R);
    const SInv = inv2x2(S);
    const zPred = mult(H, [[state.x[0]], [state.x[1]], [state.x[2]], [state.x[3]]]).map(r => r[0]);
    
    return z.map(zi => {
      const diff = zi.map((val, i) => val - zPred[i]);
      const dist = diff.reduce((sum, di, i) => 
        sum + di * diff.reduce((s, dj, j) => s + SInv[i][j] * dj, 0), 0);
      return dist < gatingSize;
    });
  }

  static momentMatching(weights: number[], states: GaussianState[]): GaussianState {
    if (states.length === 1) return { x: [...states[0].x], P: states[0].P.map(r => [...r]) };
    
    const [normWeights] = normalizeLogWeights(weights);
    const w = normWeights.map(Math.exp);
    
    const x = states[0].x.map((_, i) => 
      states.reduce((sum, s, j) => sum + w[j] * s.x[i], 0)
    );
    
    const P = states[0].P.map(row => row.map(() => 0));
    for (let i = 0; i < states.length; i++) {
      const diff = states[i].x.map((xi, j) => xi - x[j]);
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          P[r][c] += w[i] * (states[i].P[r][c] + diff[r] * diff[c]);
        }
      }
    }
    
    return { x, P };
  }
}

// ============================================================================
// MODELS
// ============================================================================

const createCVMotionModel = (T: number, sigma: number): MotionModel => ({
  d: 4,
  F: [[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]],
  Q: [
    [sigma * sigma * T ** 4 / 4, 0, sigma * sigma * T ** 3 / 2, 0],
    [0, sigma * sigma * T ** 4 / 4, 0, sigma * sigma * T ** 3 / 2],
    [sigma * sigma * T ** 3 / 2, 0, sigma * sigma * T * T, 0],
    [0, sigma * sigma * T ** 3 / 2, 0, sigma * sigma * T * T]
  ]
});

const createCVMeasModel = (sigma: number): MeasModel => ({
  d: 2,
  H: [[1, 0, 0, 0], [0, 1, 0, 0]],
  R: [[sigma * sigma, 0], [0, sigma * sigma]]
});

// ============================================================================
// DATA GENERATION
// ============================================================================

const generateObjectData = (gt: GroundTruth, motion: MotionModel): ObjectData => {
  const X: Vector[][] = Array(gt.K).fill(0).map(() => []);
  const N: number[] = Array(gt.K).fill(0);
  
  for (let i = 0; i < gt.nbirths; i++) {
    let state = [...gt.xstart[i]];
    for (let k = gt.tbirth[i] - 1; k < Math.min(gt.tdeath[i], gt.K); k++) {
      const predicted = mult(motion.F, [[state[0]], [state[1]], [state[2]], [state[3]]]).map(r => r[0]);
      state = predicted;
      X[k].push(state);
      N[k]++;
    }
  }
  
  return { X, N };
};

const generateMeasurements = (
  objectData: ObjectData,
  P_D: number,
  lambda_c: number,
  range: number[][],
  measModel: MeasModel
): Vector[][] => {
  const K = objectData.X.length;
  const measurements: Vector[][] = [];
  
  for (let k = 0; k < K; k++) {
    const meas: Vector[] = [];
    
    for (const state of objectData.X[k]) {
      if (Math.random() <= P_D) {
        const z = mult(measModel.H, [[state[0]], [state[1]], [state[2]], [state[3]]]).map(r => r[0]);
        const noise = [
          Math.sqrt(measModel.R[0][0]) * sampleStandardNormal(),
          Math.sqrt(measModel.R[1][1]) * sampleStandardNormal()
        ];
        meas.push([z[0] + noise[0], z[1] + noise[1]]);
      }
    }
    
    const N_c = samplePoisson(lambda_c);
    for (let i = 0; i < N_c; i++) {
      meas.push([
        range[0][0] + Math.random() * (range[0][1] - range[0][0]),
        range[1][0] + Math.random() * (range[1][1] - range[1][0])
      ]);
    }
    
    measurements.push(meas);
  }
  
  return measurements;
};

// ============================================================================
// CORRECTED PMBM FILTER
// ============================================================================

class PMBMFilter {
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
    const lastDeathWeight = bern.w_death[bern.w_death.length - 1];
    const lNoDetect = bern.r * (1 - P_D * lastDeathWeight);
    const likUndetected = 1 - bern.r + lNoDetect;
    const denom = 1 - lastDeathWeight * P_D;
    
    updated.r = likUndetected > 0 ? lNoDetect / likUndetected : 0;
    updated.w_death = [
      ...bern.w_death.slice(0, -1),
      lastDeathWeight * (1 - P_D)
    ].map(weight => weight / denom);
    
    return { bern: updated, logLik: Math.log(likUndetected) };
  }

  private bernoulliDetectedLikelihood(
    bern: Bernoulli,
    measurement: Vector,
    measModel: MeasModel,
    P_D: number
  ): number {
    const lastDeathWeight = bern.w_death[bern.w_death.length - 1];
    return (
      GaussianDensity.predictedLikelihood(bern.state, measurement, measModel.H, measModel.R) +
      Math.log(P_D * bern.r * lastDeathWeight)
    );
  }

  private bernoulliDetectedUpdateState(
    bern: Bernoulli,
    measurement: Vector,
    measModel: MeasModel
  ): Bernoulli {
    return {
      r: 1,
      state: GaussianDensity.update(bern.state, measurement, measModel.H, measModel.R),
      t_birth: bern.t_birth,
      t_death: [bern.t_death[bern.t_death.length - 1]],
      w_death: [1]
    };
  }

  private pppDetectedUpdate(
    k: number,
    gatingRow: boolean[],
    measurement: Vector,
    measModel: MeasModel,
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
      GaussianDensity.update(this.params.PPP.states[idx], measurement, measModel.H, measModel.R)
    );
    const updatedWeights = gatedIndices.map(idx =>
      this.params.PPP.w[idx] +
      GaussianDensity.predictedLikelihood(this.params.PPP.states[idx], measurement, measModel.H, measModel.R) +
      Math.log(P_D)
    );
    
    const [normWeights, logNumerator] = normalizeLogWeights(updatedWeights);
    const logLik = logSumExp([logNumerator, Math.log(clutterIntensity)]);
    
    return {
      bern: {
        r: Math.exp(logNumerator - logLik),
        state: GaussianDensity.momentMatching(normWeights, updatedStates),
        t_birth: k,
        t_death: [k],
        w_death: [1]
      },
      logLik
    };
  }

  initialize(birthModel: { w: number; x: Vector; P: Matrix }[]) {
    this.params.PPP.w = birthModel.map(b => Math.log(b.w));
    this.params.PPP.states = birthModel.map(b => this.cloneState({ x: b.x, P: b.P }));
    this.params.MBM = { w: [], ht: [], tt: [] };
  }
  
  predict(P_S: number, motion: MotionModel, birthModel: { w: number; x: Vector; P: Matrix }[], r_min: number) {
    this.params.PPP.w = this.params.PPP.w.map(weight => weight + Math.log(P_S));
    this.params.PPP.states = this.params.PPP.states.map(state =>
      GaussianDensity.predict(state, motion.F, motion.Q)
    );
    
    this.params.PPP.w.push(...birthModel.map(b => Math.log(b.w)));
    this.params.PPP.states.push(
      ...birthModel.map(b => this.cloneState({ x: b.x, P: b.P }))
    );
    
    this.params.MBM.tt.forEach(tree => {
      tree.forEach(bern => {
        const lastDeathWeight = bern.w_death[bern.w_death.length - 1];
        if (lastDeathWeight >= r_min) {
          bern.state = GaussianDensity.predict(bern.state, motion.F, motion.Q);
          bern.t_death = [...bern.t_death, bern.t_death[bern.t_death.length - 1] + 1];
          bern.w_death = [
            ...bern.w_death.slice(0, -1),
            lastDeathWeight * (1 - P_S),
            lastDeathWeight * P_S
          ];
        }
      });
    });
  }
  
  update(
    k: number,
    z: Vector[],
    measModel: MeasModel,
    P_D: number,
    clutter_intensity: number,
    gatingSize: number,
    w_min: number,
    M: number
  ) {
    const measurementCount = z.length;
    const pppCount = this.params.PPP.states.length;
    const used_meas_u = Array(measurementCount).fill(false);
    const gating_matrix_u = Array.from({ length: measurementCount }, () =>
      Array(pppCount).fill(false)
    );
    
    for (let pppIdx = 0; pppIdx < pppCount; pppIdx++) {
      const inGate = GaussianDensity.ellipsoidalGating(
        this.params.PPP.states[pppIdx],
        z,
        measModel.H,
        measModel.R,
        gatingSize
      );
      inGate.forEach((isInGate, measIdx) => {
        gating_matrix_u[measIdx][pppIdx] = isInGate;
        used_meas_u[measIdx] = used_meas_u[measIdx] || isInGate;
      });
    }
    
    const n_tt = this.params.MBM.tt.length;
    const used_meas_d = Array(measurementCount).fill(false);
    const gating_matrix_d: boolean[][][] = Array.from({ length: n_tt }, () => []);
    
    for (let treeIdx = 0; treeIdx < n_tt; treeIdx++) {
      const numHypo = this.params.MBM.tt[treeIdx].length;
      gating_matrix_d[treeIdx] = Array.from({ length: measurementCount }, () =>
        Array(numHypo).fill(false)
      );
      for (let hypoIdx = 0; hypoIdx < numHypo; hypoIdx++) {
        const inGate = GaussianDensity.ellipsoidalGating(
          this.params.MBM.tt[treeIdx][hypoIdx].state,
          z,
          measModel.H,
          measModel.R,
          gatingSize
        );
        inGate.forEach((isInGate, measIdx) => {
          gating_matrix_d[treeIdx][measIdx][hypoIdx] = isInGate;
          used_meas_d[measIdx] = used_meas_d[measIdx] || isInGate;
        });
      }
    }
    
    const used_meas_u_not_d = used_meas_u.map((usedU, measIdx) => usedU && !used_meas_d[measIdx]);
    const detectedMeasurementIndices = z
      .map((_, measIdx) => measIdx)
      .filter(measIdx => used_meas_d[measIdx]);
    const z_d = detectedMeasurementIndices.map(measIdx => z[measIdx]);
    const m = z_d.length;
    
    const gating_matrix_d_used = gating_matrix_d.map(treeMatrix =>
      detectedMeasurementIndices.map(measIdx => treeMatrix[measIdx] ?? [])
    );
    const gating_matrix_ud = detectedMeasurementIndices.map(measIdx => gating_matrix_u[measIdx]);
    
    const n_tt_upd = n_tt + m;
    const hypoTable: (Bernoulli | null)[][] = Array.from({ length: n_tt_upd }, () => []);
    const likTable: number[][][] = Array.from({ length: n_tt }, () => []);
    
    for (let treeIdx = 0; treeIdx < n_tt; treeIdx++) {
      const numHypo = this.params.MBM.tt[treeIdx].length;
      likTable[treeIdx] = Array.from({ length: numHypo }, () => Array(m + 1).fill(-Infinity));
      hypoTable[treeIdx] = Array(numHypo * (m + 1)).fill(null);
      
      for (let hypoIdx = 0; hypoIdx < numHypo; hypoIdx++) {
        const bern = this.params.MBM.tt[treeIdx][hypoIdx];
        const missedDetection = this.bernoulliUndetectedUpdate(bern, P_D);
        hypoTable[treeIdx][hypoIdx * (m + 1)] = missedDetection.bern;
        likTable[treeIdx][hypoIdx][0] = missedDetection.logLik;
        
        for (let measIdx = 0; measIdx < m; measIdx++) {
          if (gating_matrix_d_used[treeIdx][measIdx]?.[hypoIdx]) {
            likTable[treeIdx][hypoIdx][measIdx + 1] = this.bernoulliDetectedLikelihood(
              bern,
              z_d[measIdx],
              measModel,
              P_D
            );
            hypoTable[treeIdx][hypoIdx * (m + 1) + measIdx + 1] =
              this.bernoulliDetectedUpdateState(bern, z_d[measIdx], measModel);
          }
        }
      }
    }
    
    const logClutterIntensity = Math.log(clutter_intensity);
    const lik_new = Array(m).fill(logClutterIntensity);
    const used_meas_ud = Array(m).fill(false);
    
    for (let measIdx = 0; measIdx < m; measIdx++) {
      const detectedFromPPP = this.pppDetectedUpdate(
        k,
        gating_matrix_ud[measIdx] ?? [],
        z_d[measIdx],
        measModel,
        P_D,
        clutter_intensity
      );
      if (detectedFromPPP.bern) {
        hypoTable[n_tt + measIdx] = [detectedFromPPP.bern];
        lik_new[measIdx] = detectedFromPPP.logLik;
        used_meas_ud[measIdx] = true;
      }
    }
    
    const L2 = Array.from({ length: m }, (_, rowIdx) =>
      Array.from({ length: m }, (_, colIdx) => (rowIdx === colIdx ? -lik_new[rowIdx] : Infinity))
    );
    
    let w_upd: number[] = [];
    let ht_upd: number[][] = [];
    const H = this.params.MBM.w.length;
    
    if (H === 0) {
      const initialHT = Array(n_tt_upd).fill(0);
      for (let measIdx = 0; measIdx < m; measIdx++) {
        if (used_meas_ud[measIdx]) {
          initialHT[n_tt + measIdx] = 1;
        }
      }
      w_upd = [0];
      ht_upd = [initialHT];
    } else {
      for (let globalIdx = 0; globalIdx < H; globalIdx++) {
        const currentHT = this.params.MBM.ht[globalIdx] ?? [];
        const L1 = Array.from({ length: m }, () => Array(n_tt).fill(Infinity));
        let likTemp = 0;
        
        for (let treeIdx = 0; treeIdx < n_tt; treeIdx++) {
          const hypoIdx = currentHT[treeIdx] ?? 0;
          if (hypoIdx !== 0) {
            const localHypoIdx = hypoIdx - 1;
            const missedLik = likTable[treeIdx][localHypoIdx][0];
            likTemp += missedLik;
            for (let measIdx = 0; measIdx < m; measIdx++) {
              const detectedLik = likTable[treeIdx][localHypoIdx][measIdx + 1];
              L1[measIdx][treeIdx] = Number.isFinite(detectedLik)
                ? -(detectedLik - missedLik)
                : Infinity;
            }
          }
        }
        
        if (m === 0) {
          const htRow = currentHT.map(hypoIdx =>
            hypoIdx > 0 ? (hypoIdx - 1) * (m + 1) + 1 : 0
          );
          w_upd.push(likTemp + this.params.MBM.w[globalIdx]);
          ht_upd.push(htRow);
          continue;
        }
        
        const costMatrix = L1.map((row, rowIdx) => [...row, ...L2[rowIdx]]);
        const numAssignments = Math.max(
          1,
          Math.ceil(Math.exp(this.params.MBM.w[globalIdx]) * M)
        );
        const assignmentSolutions = kBestAssignments(costMatrix, numAssignments);
        
        assignmentSolutions.forEach(solution => {
          const colToRow = new Map<number, number>();
          solution.assignment.forEach((colIdx, rowIdx) => {
            if (colIdx >= 0) {
              colToRow.set(colIdx, rowIdx);
            }
          });
          
          const htRow = Array(n_tt_upd).fill(0);
          for (let treeIdx = 0; treeIdx < n_tt; treeIdx++) {
            const hypoIdx = currentHT[treeIdx] ?? 0;
            if (hypoIdx !== 0) {
              const assignedMeasurement = colToRow.get(treeIdx);
              htRow[treeIdx] =
                assignedMeasurement === undefined
                  ? (hypoIdx - 1) * (m + 1) + 1
                  : (hypoIdx - 1) * (m + 1) + assignedMeasurement + 2;
            }
          }
          for (let measIdx = 0; measIdx < m; measIdx++) {
            const assignedMeasurement = colToRow.get(n_tt + measIdx);
            if (assignedMeasurement !== undefined && used_meas_ud[assignedMeasurement]) {
              htRow[n_tt + measIdx] = 1;
            }
          }
          
          w_upd.push(-solution.cost + likTemp + this.params.MBM.w[globalIdx]);
          ht_upd.push(htRow);
        });
      }
    }
    
    const extraMeasurementIndices = z
      .map((_, measIdx) => measIdx)
      .filter(measIdx => used_meas_u_not_d[measIdx]);
    
    extraMeasurementIndices.forEach(measIdx => {
      const detectedFromPPP = this.pppDetectedUpdate(
        k,
        gating_matrix_u[measIdx],
        z[measIdx],
        measModel,
        P_D,
        clutter_intensity
      );
      hypoTable.push(detectedFromPPP.bern ? [detectedFromPPP.bern] : []);
    });
    
    if (extraMeasurementIndices.length > 0) {
      ht_upd = ht_upd.map(row => [...row, ...Array(extraMeasurementIndices.length).fill(1)]);
    }
    
    this.params.PPP.w = this.params.PPP.w.map(weight => weight + Math.log(1 - P_D));
    
    const finiteGlobalHypotheses = w_upd
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

    w_upd = normalizedGlobalHypotheses.map(({ weight }) => weight);
    ht_upd = normalizedGlobalHypotheses.map(({ idx }) => [...ht_upd[idx]]);

    if (w_upd.length > M) {
      const capped = w_upd
        .map((weight, idx) => ({ weight, idx }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, M);
      w_upd = capped.map(({ weight }) => weight);
      ht_upd = capped.map(({ idx }) => [...ht_upd[idx]]);
    }
    [w_upd] = normalizeLogWeights(w_upd);
    
    const totalTreeCount = n_tt_upd + extraMeasurementIndices.length;
    const keepTreeMask = Array(totalTreeCount).fill(false);
    ht_upd.forEach(row => {
      for (let treeIdx = 0; treeIdx < totalTreeCount; treeIdx++) {
        keepTreeMask[treeIdx] = keepTreeMask[treeIdx] || (row[treeIdx] ?? 0) > 0;
      }
    });
    
    const filteredHypoTable = hypoTable.filter((_, treeIdx) => keepTreeMask[treeIdx]);
    const filteredHT = ht_upd.map(row => row.filter((_, treeIdx) => keepTreeMask[treeIdx]));
    const newTrees = filteredHypoTable.map((treeHypotheses, treeIdx) => {
      const referencedHypotheses = stableUnique(
        filteredHT.map(row => row[treeIdx]).filter(localIdx => localIdx > 0)
      );
      return referencedHypotheses
        .map(localIdx => treeHypotheses[localIdx - 1])
        .filter((bern): bern is Bernoulli => bern !== null)
        .map(bern => this.cloneBernoulli(bern));
    });
    
    const filteredTreeCount = filteredHT.length > 0 ? filteredHT[0].length : 0;
    for (let treeIdx = 0; treeIdx < filteredTreeCount; treeIdx++) {
      const activeIndices = stableUnique(
        filteredHT.map(row => row[treeIdx]).filter(localIdx => localIdx > 0)
      );
      const indexMap = new Map(activeIndices.map((localIdx, idx) => [localIdx, idx + 1]));
      filteredHT.forEach(row => {
        if ((row[treeIdx] ?? 0) > 0) {
          row[treeIdx] = indexMap.get(row[treeIdx]) ?? 0;
        }
      });
    }
    
    this.params.MBM.w = w_upd;
    this.params.MBM.ht = filteredHT.length > 0 ? filteredHT : [];
    this.params.MBM.tt = newTrees;
  }
  
  prune(r_min: number, w_min: number) {
    for (let treeIdx = 0; treeIdx < this.params.MBM.tt.length; treeIdx++) {
      const tree = this.params.MBM.tt[treeIdx];
      const prunedIndices: number[] = [];
      const keptBernoullis: Bernoulli[] = [];
      
      tree.forEach((bern, localIdx) => {
        if (bern.r < r_min) {
          prunedIndices.push(localIdx + 1);
        } else {
          keptBernoullis.push(bern);
        }
      });
      
      this.params.MBM.tt[treeIdx] = keptBernoullis;
      prunedIndices.forEach(prunedIdx => {
        this.params.MBM.ht.forEach(row => {
          if (row[treeIdx] === prunedIdx) {
            row[treeIdx] = 0;
          }
        });
      });
    }
    
    if (this.params.MBM.ht.length > 0) {
      const keepTreeMask = this.params.MBM.ht[0].map((_, treeIdx) =>
        this.params.MBM.ht.some(row => row[treeIdx] !== 0)
      );
      this.params.MBM.ht = this.params.MBM.ht.map(row =>
        row.filter((_, treeIdx) => keepTreeMask[treeIdx])
      );
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
        const key = row.join(",");
        const duplicateIndices = rowGroups.get(key) ?? [];
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
  
  extractEstimates(r_threshold: number): Vector[] {
    const estimates: Vector[] = [];
    
    if (this.params.MBM.w.length > 0 && this.params.MBM.ht.length > 0) {
      const bestGlobalHypothesisIdx = this.params.MBM.w.indexOf(Math.max(...this.params.MBM.w));
      const bestHypothesis = this.params.MBM.ht[bestGlobalHypothesisIdx] ?? [];
      
      for (let treeIdx = 0; treeIdx < bestHypothesis.length && treeIdx < this.params.MBM.tt.length; treeIdx++) {
        const localHypoIdx = bestHypothesis[treeIdx];
        if (localHypoIdx > 0 && localHypoIdx <= this.params.MBM.tt[treeIdx].length) {
          const bern = this.params.MBM.tt[treeIdx][localHypoIdx - 1];
          if (bern.r >= r_threshold) {
            estimates.push([...bern.state.x]);
          }
        }
      }
    }
    
    return estimates;
  }
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PMBMFilterSimulation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [timeStep, setTimeStep] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  
  const [visibility, setVisibility] = useState({
    groundTruth: true,
    measurements: true,
    estimates: true
  });
  
  const [params, setParams] = useState({
    P_D: 0.98,
    lambda_c: 10,
    P_S: 0.99,
    sigma_q: 5,
    sigma_r: 10
  });
  
  const [simulation, setSimulation] = useState<{
    groundTruth: ObjectData;
    measurements: Vector[][];
    estimates: Vector[][];
    K: number;
  } | null>(null);
  
  const initializeSimulation = () => {
    const K = 100;
    const groundTruth: GroundTruth = {
      nbirths: 12,
      K,
      xstart: [
        [0, 0, 0, -10],
        [400, -600, -10, 5],
        [-800, -200, 20, -5],
        [400, -600, -7, -4],
        [400, -600, -2.5, 10],
        [0, 0, 7.5, -5],
        [-800, -200, 12, 7],
        [-200, 800, 15, -10],
        [-800, -200, 3, 15],
        [-200, 800, -3, -15],
        [0, 0, -20, -15],
        [-200, 800, 15, -5]
      ],
      tbirth: [1, 1, 1, 20, 20, 20, 40, 40, 60, 60, 80, 80],
      tdeath: [70, K + 1, 70, K + 1, K + 1, K + 1, K + 1, K + 1, K + 1, K + 1, K + 1, K + 1]
    };
    
    const motion = createCVMotionModel(1, params.sigma_q);
    const meas = createCVMeasModel(params.sigma_r);
    const range = [[-1000, 1000], [-1000, 1000]];
    
    const objectData = generateObjectData(groundTruth, motion);
    const measurements = generateMeasurements(
      objectData,
      params.P_D,
      params.lambda_c,
      range,
      meas
    );
    
    const birthModel = [
      { w: 0.03, x: [0, 0, 0, 0], P: scale(400, identity(4)) },
      { w: 0.03, x: [400, -600, 0, 0], P: scale(400, identity(4)) },
      { w: 0.03, x: [-800, -200, 0, 0], P: scale(400, identity(4)) },
      { w: 0.03, x: [-200, 800, 0, 0], P: scale(400, identity(4)) }
    ];
    
    const filter = new PMBMFilter();
    filter.initialize(birthModel);
    
    const estimates: Vector[][] = [];
    const gatingSize = 13.815510557964274;
    const w_min = Math.log(0.0001);
    const M = 100;
    const r_min = 0.0001;
    const estimateThreshold = 0.5;
    const clutterIntensity =
      params.lambda_c /
      ((range[0][1] - range[0][0]) * (range[1][1] - range[1][0]));
    
    for (let k = 0; k < K; k++) {
      filter.update(k + 1, measurements[k], meas, params.P_D, clutterIntensity, gatingSize, w_min, M);
      estimates.push(filter.extractEstimates(estimateThreshold));
      filter.prune(r_min, w_min);
      filter.predict(params.P_S, motion, birthModel, r_min);
    }
    
    setSimulation({ groundTruth: objectData, measurements, estimates, K });
    setTimeStep(0);
    setIsRunning(true);
  };
  
  useEffect(() => {
    const timer = window.setTimeout(() => {
      initializeSimulation();
    }, 0);

    return () => {
      window.clearTimeout(timer);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  
  useEffect(() => {
    if (!isRunning || !simulation) return;
    
    const interval = setInterval(() => {
      setTimeStep(t => {
        if (t >= simulation.K - 1) {
          setIsRunning(false);
          return t;
        }
        return t + 1;
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [isRunning, simulation]);
  
  useEffect(() => {
    if (!simulation || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const scale = 0.4;
    const offsetX = width / 2;
    const offsetY = height / 2;
    
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, width, height);
    
    // Grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = -1000; i <= 1000; i += 200) {
      ctx.beginPath();
      ctx.moveTo(offsetX + i * scale, 0);
      ctx.lineTo(offsetX + i * scale, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, offsetY + i * scale);
      ctx.lineTo(width, offsetY + i * scale);
      ctx.stroke();
    }
    
    // Axes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(offsetX, 0);
    ctx.lineTo(offsetX, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, offsetY);
    ctx.lineTo(width, offsetY);
    ctx.stroke();
    
    // Measurements
    if (visibility.measurements) {
      ctx.fillStyle = 'rgba(249, 115, 22, 0.6)';
      for (const z of simulation.measurements[timeStep]) {
        ctx.beginPath();
        ctx.arc(offsetX + z[0] * scale, offsetY - z[1] * scale, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    
    // Ground truth
    if (visibility.groundTruth) {
      ctx.fillStyle = '#38bdf8';
      for (let k = 0; k <= timeStep; k++) {
        const alpha = k === timeStep ? 1 : 0.3;
        ctx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
        for (const state of simulation.groundTruth.X[k]) {
          ctx.beginPath();
          ctx.arc(offsetX + state[0] * scale, offsetY - state[1] * scale, 5, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
    
    // Estimates
    if (visibility.estimates) {
      for (let k = 0; k <= timeStep; k++) {
        const alpha = k === timeStep ? 1 : 0.22;
        const radius = k === timeStep ? 8 : 5;
        const arm = k === timeStep ? 12 : 7;
        const stroke = k === timeStep ? 3 : 2;
        ctx.strokeStyle = `rgba(239, 68, 68, ${alpha})`;
        ctx.fillStyle = `rgba(239, 68, 68, ${Math.min(0.8, alpha)})`;
        ctx.lineWidth = stroke;

        for (const est of simulation.estimates[k]) {
          const x = offsetX + est[0] * scale;
          const y = offsetY - est[1] * scale;
          
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.fill();
          
          ctx.beginPath();
          ctx.moveTo(x - arm, y);
          ctx.lineTo(x + arm, y);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.moveTo(x, y - arm);
          ctx.lineTo(x, y + arm);
          ctx.stroke();
        }
      }
    }
    
    // Legend
    ctx.font = '14px sans-serif';
    let legendY = 25;
    
    if (visibility.groundTruth) {
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(30, legendY, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText('Ground Truth', 50, legendY + 4);
      legendY += 25;
    }
    
    if (visibility.measurements) {
      ctx.fillStyle = '#f97316';
      ctx.beginPath();
      ctx.arc(30, legendY, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText('Measurements', 50, legendY + 4);
      legendY += 25;
    }
    
    if (visibility.estimates) {
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(30, legendY, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(22, legendY);
      ctx.lineTo(38, legendY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(30, legendY - 8);
      ctx.lineTo(30, legendY + 8);
      ctx.stroke();
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText('PMBM Estimates', 50, legendY + 4);
    }
    
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`Time: ${timeStep + 1}/${simulation.K}`, width - 120, 30);
  }, [simulation, timeStep, visibility]);
  
  return (
    <main className="page-shell demo-shell">
      <DemoShellStyles />

      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Multi-Object Tracking / PMBM</p>
          <h1>PMBM Filter Simulation</h1>
          <p className="lede">
            Poisson Multi-Bernoulli Mixture filtering for multi-object tracking with corrected
            hypothesis handling, estimate extraction, and assignment logic.
          </p>
        </div>

        <div className="control-panel">
          <div className="button-row">
            <button
              type="button"
              onClick={() => setIsRunning(!isRunning)}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-slate-950 transition-colors hover:bg-cyan-400 disabled:bg-slate-800 disabled:text-slate-500"
              disabled={!simulation}
            >
              {isRunning ? <Pause size={20} /> : <Play size={20} />}
              {isRunning ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              onClick={() => {
                setIsRunning(false);
                setTimeStep(0);
              }}
              className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 transition-colors hover:border-slate-600 hover:bg-slate-800 disabled:border-slate-800 disabled:bg-slate-900/60 disabled:text-slate-500"
              disabled={!simulation}
            >
              <RotateCcw size={20} />
            </button>
            <button
              type="button"
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 transition-colors hover:border-slate-600 hover:bg-slate-800"
            >
              <Settings size={20} />
            </button>
            <button
              type="button"
              onClick={() => setShowInfo(!showInfo)}
              className="flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-400/12 px-3 py-2 text-cyan-100 transition-colors hover:bg-cyan-400/18"
            >
              <Info size={18} />
              {showInfo ? 'Hide Info' : 'Show Info'}
            </button>
          </div>

          <div className="status-strip">
            <span className="inline-flex rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-2 text-sm text-cyan-50">
              Step {timeStep + 1}
            </span>
            {simulation && (
              <>
                <span className="inline-flex rounded-full border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-200">
                  {simulation.groundTruth.N[timeStep]} true objects
                </span>
                <span className="inline-flex rounded-full border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-200">
                  {simulation.estimates[timeStep].length} estimates
                </span>
              </>
            )}
          </div>

          <p className="detail-copy">
            Toggle layers from the sidebar to compare cluttered measurements, ground truth, and the
            extracted PMBM track set at each time step.
          </p>
        </div>
      </section>

      <section className="content-grid">
        <article className="canvas-card">
          <div className="section-head">
            <div>
              <p className="mini-label">Simulation</p>
              <h2>Ground truth, clutter, and PMBM estimates</h2>
            </div>
            <p className="annotation">
              The canvas overlays detections, false alarms, and extracted tracks on the same dark
              field so you can compare associations frame by frame.
            </p>
          </div>

          <div className="viz-surface">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full"
            />
          </div>
        </article>

        <aside className="sidebar">
          {showInfo && (
            <article className="metric-card">
              <div className="section-head compact">
                <div>
                  <p className="mini-label">About PMBM</p>
                  <h2>Filter structure</h2>
                </div>
              </div>
              <div className="grid gap-4 text-sm text-slate-300 md:grid-cols-2">
                <div>
                  <strong>PPP (Poisson Point Process):</strong>
                  <p className="mt-2">Models undetected and newly born objects.</p>
                  <p>Its intensity captures the expected object density.</p>
                </div>
                <div>
                  <strong>MBM (Multi-Bernoulli Mixture):</strong>
                  <p className="mt-2">Tracks confirmed objects and competing hypotheses.</p>
                  <p>Each Bernoulli represents one potential target.</p>
                </div>
                <div>
                  <strong>Key features:</strong>
                  <p className="mt-2">Handles object birth and death.</p>
                  <p>Maintains data-association uncertainty over time.</p>
                </div>
                <div>
                  <strong>Filter loop:</strong>
                  <p className="mt-2">Predict existing tracks.</p>
                  <p>Update with measurements, then extract tracks with r ≥ 0.5.</p>
                </div>
              </div>
            </article>
          )}

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Visibility</p>
                <h2>Layer controls</h2>
              </div>
            </div>
            <div className="space-y-2">
              <button
                type="button"
                onClick={() => setVisibility(v => ({ ...v, groundTruth: !v.groundTruth }))}
                className="flex w-full items-center justify-between rounded-2xl border border-slate-800 bg-slate-950/80 px-3 py-2 transition-colors hover:border-slate-700 hover:bg-slate-900"
              >
                <span className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded-full bg-cyan-400"></div>
                  <span className="text-sm text-slate-100">Ground Truth</span>
                </span>
                {visibility.groundTruth ? <Eye size={16} className="text-cyan-200" /> : <EyeOff size={16} className="text-slate-500" />}
              </button>

              <button
                type="button"
                onClick={() => setVisibility(v => ({ ...v, measurements: !v.measurements }))}
                className="flex w-full items-center justify-between rounded-2xl border border-slate-800 bg-slate-950/80 px-3 py-2 transition-colors hover:border-slate-700 hover:bg-slate-900"
              >
                <span className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded-full bg-orange-500"></div>
                  <span className="text-sm text-slate-100">Measurements</span>
                </span>
                {visibility.measurements ? <Eye size={16} className="text-cyan-200" /> : <EyeOff size={16} className="text-slate-500" />}
              </button>

              <button
                type="button"
                onClick={() => setVisibility(v => ({ ...v, estimates: !v.estimates }))}
                className="flex w-full items-center justify-between rounded-2xl border border-slate-800 bg-slate-950/80 px-3 py-2 transition-colors hover:border-slate-700 hover:bg-slate-900"
              >
                <span className="flex items-center gap-2">
                  <div className="h-4 w-4 rounded-full bg-red-500"></div>
                  <span className="text-sm text-slate-100">PMBM Estimates</span>
                </span>
                {visibility.estimates ? <Eye size={16} className="text-cyan-200" /> : <EyeOff size={16} className="text-slate-500" />}
              </button>
            </div>
          </article>

          {showSettings && (
            <article className="metric-card">
              <div className="section-head compact">
                <div>
                  <p className="mini-label">Parameters</p>
                  <h2>Scenario tuning</h2>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-sm font-medium text-slate-300">
                    Detection Prob (P_D): {params.P_D.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.01"
                    value={params.P_D}
                    onChange={e => setParams({ ...params, P_D: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium text-slate-300">
                    Clutter Rate (λ_c): {params.lambda_c}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="1"
                    value={params.lambda_c}
                    onChange={e => setParams({ ...params, lambda_c: parseInt(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-sm font-medium text-slate-300">
                    Motion Noise (σ_q): {params.sigma_q}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="0.5"
                    value={params.sigma_q}
                    onChange={e => setParams({ ...params, sigma_q: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <button
                  type="button"
                  onClick={initializeSimulation}
                  className="w-full rounded-xl bg-cyan-500 px-4 py-2 text-slate-950 transition-colors hover:bg-cyan-400"
                >
                  Generate New Scenario
                </button>
              </div>
            </article>
          )}

          {simulation && (
            <article className="metric-card">
              <div className="section-head compact">
                <div>
                  <p className="mini-label">Statistics</p>
                  <h2>Frame {timeStep + 1}</h2>
                </div>
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">True Objects</span>
                  <span className="font-mono text-slate-100">{simulation.groundTruth.N[timeStep]}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Measurements</span>
                  <span className="font-mono text-slate-100">{simulation.measurements[timeStep].length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">PMBM Estimates</span>
                  <span className={`font-mono font-bold ${simulation.estimates[timeStep].length > 0 ? 'text-cyan-300' : 'text-red-400'}`}>
                    {simulation.estimates[timeStep].length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Total Tracked</span>
                  <span className="font-mono text-cyan-300">
                    {simulation.estimates.filter(est => est.length > 0).length} / {simulation.K} steps
                  </span>
                </div>
              </div>
            </article>
          )}

          <article className="metric-card">
            <div className="section-head compact">
              <div>
                <p className="mini-label">Legend</p>
                <h2>Reading the canvas</h2>
              </div>
            </div>
            <ul className="notes-list text-sm text-slate-300">
              <li><strong>Cyan circles</strong> show true object positions.</li>
              <li><strong>Orange dots</strong> show measurements, including clutter.</li>
              <li><strong>Red crosses</strong> show PMBM estimates with faded history.</li>
              <li>Objects appear and disappear at different times.</li>
              <li>The filter must separate clutter from persistent targets.</li>
            </ul>
          </article>
        </aside>
      </section>
    </main>
  );
}
