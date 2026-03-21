'use client';

import dynamic from "next/dynamic";
import { useState } from "react";
import { Activity, BarChart3, ChevronRight, Leaf, Map, Radar, type LucideIcon } from "lucide-react";

type DemoKey = "lawnmower" | "planner" | "pmbm" | "financial" | "particle" | "kalman";

type DemoConfig = {
  id: DemoKey;
  title: string;
  strapline: string;
  description: string;
  Icon: LucideIcon;
};

const PMBMSimulation = dynamic(() => import("./pmbm-filter-corrected-tsx"), {
  ssr: false,
  loading: () => <LoadingState label="Loading PMBM simulation" />,
});

const FinancialTracker = dynamic(() => import("./financial-kalman-tracker"), {
  ssr: false,
  loading: () => <LoadingState label="Loading financial tracker" />,
});

const LawnmowerBBR = dynamic(() => import("./lawnmower-bbr"), {
  ssr: false,
  loading: () => <LoadingState label="Loading lawnmower simulation" />,
});

const PathPlanner = dynamic(() => import("./path-planner"), {
  ssr: false,
  loading: () => <LoadingState label="Loading path planner" />,
});

const ParticleFilterMap = dynamic(() => import("./particle-filter-map"), {
  ssr: false,
  loading: () => <LoadingState label="Loading particle filter map" />,
});

const KalmanFilterViz = dynamic(() => import("./kalman-filter-viz-fixed"), {
  ssr: false,
  loading: () => <LoadingState label="Loading Kalman filter visualization" />,
});

const demos: DemoConfig[] = [
  {
    id: "planner",
    title: "Path Planner",
    strapline: "Grid planner with local vehicle model",
    description:
      "Run the browser path-planning demo with its Dijkstra planner, lookahead controller, and local vehicle dynamics.",
    Icon: Map,
  },
  {
    id: "financial",
    title: "Financial Tracker",
    strapline: "PMBM-driven trading",
    description:
      "Explore the financial tracker with the PMBM estimate stream and corrected trade execution and exit logic.",
    Icon: BarChart3,
  },
  {
    id: "lawnmower",
    title: "Lawnmower",
    strapline: "BEHAVIOUR-BASED ROBOTICS",
    description:
      "Run the behavior-based lawnmower assignment with a local grass, weather, battery, and vehicle simulation.",
    Icon: Leaf,
  },
  {
    id: "particle",
    title: "Particle Filter Map",
    strapline: "Interactive road-map tracker",
    description:
      "Sketch a route across the map, initialize the filter from known or unknown starts, and replay the particle cloud frame by frame.",
    Icon: Map,
  },
  {
    id: "pmbm",
    title: "PMBM Simulation",
    strapline: "Reference multi-object tracker",
    description:
      "Run the corrected Poisson multi-Bernoulli mixture simulation with the original tracking controls and diagnostics.",
    Icon: Activity,
  },
  {
    id: "kalman",
    title: "Kalman Filter",
    strapline: "Non-linear coordinated-turn tracker",
    description:
      "Compare the true coordinated-turn trajectory with the CKF estimate and explore how different process-noise profiles affect tracking quality.",
    Icon: Radar,
  },
];

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex h-screen items-center justify-center bg-slate-950 px-6 text-slate-200">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 px-6 py-4 text-sm shadow-2xl shadow-slate-950/40">
        {label}
      </div>
    </div>
  );
}

export default function Home() {
  const [selectedDemo, setSelectedDemo] = useState<DemoKey>("planner");

  const activeDemo = demos.find((demo) => demo.id === selectedDemo) ?? demos[0];
  const ActiveComponent =
    selectedDemo === "lawnmower"
      ? LawnmowerBBR
      : selectedDemo === "planner"
      ? PathPlanner
      : selectedDemo === "pmbm"
      ? PMBMSimulation
      : selectedDemo === "financial"
      ? FinancialTracker
      : selectedDemo === "particle"
        ? ParticleFilterMap
        : KalmanFilterViz;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 md:flex">
      <aside className="border-b border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),transparent_36%),linear-gradient(180deg,#020617_0%,#0f172a_65%,#111827_100%)] md:sticky md:top-0 md:h-screen md:w-80 md:shrink-0 md:border-b-0 md:border-r">
        <div className="flex h-full flex-col p-4 md:p-6">
          <div>
            <h1 className="mt-3 text-2xl font-semibold text-white">Robotics Portfolio</h1>
            <p className="mt-2 max-w-sm text-sm leading-6 text-slate-300">
              Choose which interactive demo to run from the menu.
            </p>
          </div>

          <nav className="mt-6 flex gap-3 overflow-x-auto pb-1 md:flex-col md:overflow-visible">
            {demos.map((demo) => {
              const isActive = demo.id === selectedDemo;
              const Icon = demo.Icon;

              return (
                <button
                  key={demo.id}
                  type="button"
                  onClick={() => setSelectedDemo(demo.id)}
                  className={`group min-w-[240px] rounded-2xl border px-4 py-4 text-left transition-all md:min-w-0 ${
                    isActive
                      ? "border-cyan-400/60 bg-cyan-400/12 shadow-lg shadow-cyan-950/30"
                      : "border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`mt-0.5 rounded-xl p-2 ${
                        isActive ? "bg-cyan-400/15 text-cyan-200" : "bg-slate-800 text-slate-300"
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <p className="font-medium text-white">{demo.title}</p>
                        <ChevronRight
                          className={`h-4 w-4 shrink-0 transition-transform ${
                            isActive ? "translate-x-0 text-cyan-200" : "text-slate-500 group-hover:translate-x-0.5"
                          }`}
                        />
                      </div>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                        {demo.strapline}
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </nav>

          <div className="mt-6 hidden rounded-3xl border border-slate-800 bg-slate-900/70 p-5 md:block">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
              Now Running
            </p>
            <h2 className="mt-2 text-lg font-semibold text-white">{activeDemo.title}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">{activeDemo.description}</p>
          </div>

          <div className="mt-auto hidden pt-6 text-xs leading-5 text-slate-400 md:block">
            The selected demo fills the workspace on the right. Switch tools without changing routes.
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 bg-slate-950">
        <ActiveComponent />
      </main>
    </div>
  );
}
