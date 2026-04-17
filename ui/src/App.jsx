import React, { useCallback, useEffect, useRef, useState } from "react";
import "./index.css";
import BearingCompass   from "./components/BearingCompass";
import IntersectionMap  from "./components/IntersectionMap";
import EventFeed        from "./components/EventFeed";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_URL       = "ws://localhost:8000/ws";
const RECONNECT_MS = 3000;
const MAX_EVENTS   = 120;

const LANE_LABELS = {
  approach_north: "North",
  approach_south: "South",
  approach_east:  "East",
  approach_west:  "West",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function beliefClass(belief) {
  if (belief > 0.65) return "high";
  if (belief > 0.30) return "mid";
  return "low";
}

function beliefTextColor(belief) {
  if (belief > 0.65) return "var(--green)";
  if (belief > 0.30) return "var(--amber)";
  return "var(--blue)";
}

let _evtId = 0;
function mkEvent(type, data = {}) {
  return { id: _evtId++, timestamp: Date.now() / 1000, type, ...data };
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  // Connection state
  const [connected, setConnected]   = useState(false);
  const wsRef      = useRef(null);
  const retryRef   = useRef(null);

  // Pipeline telemetry
  const [audioConf,    setAudioConf]    = useState(0);
  const [audioBearing, setAudioBearing] = useState(0);
  const [detected,     setDetected]     = useState(false);
  const [beliefs,      setBeliefs]      = useState({
    approach_north: 0,
    approach_south: 0,
    approach_east:  0,
    approach_west:  0,
  });
  const [phases,        setPhases]       = useState({});
  const [tlsStates,     setTlsStates]    = useState({});
  const [activeLanes,   setActiveLanes]  = useState([]);
  const [visionCount,   setVisionCount]  = useState(0);

  // Events log
  const [events, setEvents] = useState([
    mkEvent("connected", { message: "Waiting for backend connection…" }),
  ]);

  // Flash effect on preemption trigger
  const [flash, setFlash] = useState(false);

  const addEvent = useCallback((evt) => {
    setEvents(prev => [...prev.slice(-(MAX_EVENTS - 1)), evt]);
  }, []);

  // ------------------------------------------------------------------
  // WebSocket connection with auto-reconnect
  // ------------------------------------------------------------------

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      clearTimeout(retryRef.current);
      addEvent(mkEvent("connected", { message: "Backend connected" }));
    };

    ws.onmessage = (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch { return; }

      // Update telemetry
      setAudioConf(data.audio_conf    ?? 0);
      setAudioBearing(data.audio_bearing ?? 0);
      setDetected(data.audio_detected ?? false);
      setVisionCount(data.vision_count ?? 0);

      if (data.fusion_beliefs) setBeliefs(prev => ({ ...prev, ...data.fusion_beliefs }));
      if (data.fusion_phases)  setPhases(prev  => ({ ...prev, ...data.fusion_phases }));
      if (data.tls_states)     setTlsStates(prev => ({ ...prev, ...data.tls_states }));

      const active = data.active_preemptions ?? [];
      setActiveLanes(active);

      // Log preemption events
      if (data.preempt_fired && active.length > 0) {
        setFlash(true);
        setTimeout(() => setFlash(false), 800);
        active.forEach(lane => {
          addEvent(mkEvent("preempt", {
            lane,
            belief: data.fusion_beliefs?.[lane] ?? 0,
          }));
        });
      }

      // Log phase transitions
      if (data.fusion_phases) {
        Object.entries(data.fusion_phases).forEach(([lane, phase]) => {
          if (phase === "ARMED") {
            // Only emit once — deduplicate by checking last event
            setEvents(prev => {
              const last = prev[prev.length - 1];
              if (last?.type === "armed" && last?.lane === lane) return prev;
              return [...prev.slice(-(MAX_EVENTS - 1)), mkEvent("armed", { lane })];
            });
          }
        });
      }
    };

    ws.onclose = () => {
      setConnected(false);
      addEvent(mkEvent("disconnected", { message: "Backend disconnected — reconnecting…" }));
      retryRef.current = setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror = () => ws.close();
  }, [addEvent]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <div className="shell">
      {flash && <div className="preempt-flash" />}

      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <svg width={20} height={20} viewBox="0 0 20 20" fill="none">
            <circle cx={10} cy={10} r={9} stroke="var(--green)" strokeWidth={1.5} />
            <path d="M10 3 L10 10 L14 14" stroke="var(--green)" strokeWidth={1.5} strokeLinecap="round" />
          </svg>
          <h1>GREENWAVE++</h1>
          <span className="subtitle">Emergency Vehicle Preemption System</span>
        </div>

        <div className="header-right">
          <div className="mono muted" style={{ fontSize: 11 }}>
            Vision: {visionCount} det
          </div>
          <div className={`status-pill ${connected ? "online" : "offline"}`}>
            <span className={`status-dot ${connected ? "pulse" : ""}`} />
            {connected ? "SYSTEM ONLINE" : "OFFLINE"}
          </div>
        </div>
      </header>

      {/* ── Main content ─────────────────────────────────────────── */}
      <main className="main-content">

        {/* Top metrics row */}
        <div className="metrics-row">
          <MetricCard
            label="Audio Confidence"
            value={`${(audioConf * 100).toFixed(1)}%`}
            colorClass={audioConf > 0.65 ? "amber" : audioConf > 0.35 ? "blue" : ""}
          />
          <MetricCard
            label="Active Preemptions"
            value={activeLanes.length}
            colorClass={activeLanes.length > 0 ? "green" : ""}
            sub={activeLanes.map(l => LANE_LABELS[l] || l).join(", ") || "None"}
          />
          <MetricCard
            label="Peak Belief"
            value={`${(Math.max(0, ...Object.values(beliefs)) * 100).toFixed(0)}%`}
            colorClass={
              Math.max(0, ...Object.values(beliefs)) > 0.75 ? "green" :
              Math.max(0, ...Object.values(beliefs)) > 0.45 ? "amber" : ""
            }
          />
          <MetricCard
            label="TLS Controlled"
            value={Object.keys(tlsStates).length}
            colorClass="blue"
            sub="traffic lights"
          />
        </div>

        {/* Main 3-column grid */}
        <div className="main-grid" style={{ flex: 1, minHeight: 0 }}>
          <BearingCompass
            bearing    = {audioBearing}
            confidence = {audioConf}
            detected   = {detected}
          />

          <IntersectionMap
            beliefs      = {beliefs}
            phases       = {phases}
            tlsStates    = {tlsStates}
            activeLanes  = {activeLanes}
          />

          <div className="event-feed-col" style={{ minHeight: 0, overflow: "hidden" }}>
            <EventFeed events={events} />
          </div>
        </div>

        {/* Belief bars — one per lane */}
        <div className="belief-section">
          {Object.entries(beliefs).map(([laneId, belief]) => {
            const phase      = phases[laneId] || "IDLE";
            const isActive   = phase === "ACTIVE"  || activeLanes.includes(laneId);
            const isPreempt  = activeLanes.includes(laneId);

            return (
              <div
                key={laneId}
                className={`belief-bar-card ${isPreempt ? "preempting" : isActive ? "active" : ""}`}
              >
                <div className="belief-lane-name">
                  {LANE_LABELS[laneId] || laneId}
                  <span className={`phase-badge ${phase}`}>{phase}</span>
                </div>

                <div
                  className="belief-percent"
                  style={{ color: beliefTextColor(belief) }}
                >
                  {(belief * 100).toFixed(1)}%
                </div>

                <div className="belief-bar-track">
                  <div
                    className={`belief-bar-fill ${beliefClass(belief)}`}
                    style={{ width: `${belief * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricCard
// ---------------------------------------------------------------------------

function MetricCard({ label, value, colorClass = "", sub }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${colorClass}`}>{value}</div>
      {sub && <div className="metric-sub muted">{sub}</div>}
    </div>
  );
}