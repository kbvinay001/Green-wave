import React, { useEffect, useRef } from "react";

/**
 * Bird's-eye intersection map — Green Wave++
 *
 * SVG rendering of a 4-way intersection.  Each approach road is overlaid
 * with a semi-transparent belief heat tint (blue→amber→green).  TLS nodes
 * show actual signal state from the backend.  When a lane is ACTIVE
 * (preemption running), the approach road pulses green.
 *
 * Props:
 *   beliefs       { approach_north: 0.72, … }   — 0..1 per lane
 *   phases        { approach_north: "ARMED", … } — fusion phase string
 *   tlsStates     { J_N1: "green", J_N2: "red", … }
 *   activeLanes   ["approach_north"]              — lanes being preempted
 */
export default function IntersectionMap({ beliefs = {}, phases = {}, tlsStates = {}, activeLanes = [] }) {
  const size   = 360;
  const cx     = size / 2;
  const cy     = size / 2;

  const roadW  = 52;   // road width in SVG units
  const intW   = roadW * 1.4;   // intersection box half-width

  // Lane configs: name, direction (angle from center, degrees), TLS ids
  const lanes = [
    { id: "approach_north", label: "N", angle: -90, tlsIds: ["J_N1", "J_N2", "J_N3"] },
    { id: "approach_east",  label: "E", angle:   0, tlsIds: ["J_E1", "J_E2"] },
    { id: "approach_south", label: "S", angle:  90, tlsIds: ["J_S1", "J_S2", "J_S3"] },
    { id: "approach_west",  label: "W", angle: 180, tlsIds: ["J_W1", "J_W2"] },
  ];

  function beliefColor(belief, phase, active) {
    if (active) return "rgba(0, 230, 118, 0.30)";
    if (phase === "ACTIVE")  return "rgba(0, 230, 118, 0.25)";
    if (phase === "ARMED")   return "rgba(255, 171, 64, 0.28)";
    if (belief > 0.6)        return `rgba(255, 171, 64, ${belief * 0.35})`;
    if (belief > 0.3)        return `rgba(64, 196, 255, ${belief * 0.4})`;
    return `rgba(64, 196, 255, ${belief * 0.25})`;
  }

  function tlsColor(state) {
    if (state === "green")  return "#00e676";
    if (state === "yellow") return "#ffab40";
    return "#ff5252";
  }

  // Draw one approach road + TLS nodes
  function Road({ lane }) {
    const belief  = beliefs[lane.id] || 0;
    const phase   = phases[lane.id]  || "IDLE";
    const active  = activeLanes.includes(lane.id);
    const rad     = lane.angle * (Math.PI / 180);

    // Road rectangle runs from intersection edge (intW) outward to edge (cx)
    const armLen  = cx - intW - 10;

    // Road strip along the axis
    const rx = lane.angle === -90 || lane.angle === 90
      ? cx - roadW / 2
      : (lane.angle === 0 ? cx + intW : 0);
    const ry = lane.angle === -90
      ? 0
      : lane.angle === 90
      ? cy + intW
      : cy - roadW / 2;
    const rw = lane.angle === -90 || lane.angle === 90 ? roadW : armLen;
    const rh = lane.angle === -90 || lane.angle === 90 ? armLen : roadW;

    const fillColor = beliefColor(belief, phase, active);

    // TLS dot positions (evenly spaced along approach arm)
    const tlsDots = lane.tlsIds.map((id, i) => {
      const dist = intW + 15 + i * 28;
      const dx   = cx + Math.cos(rad) * dist;
      const dy   = cy + Math.sin(rad) * dist;
      const state = tlsStates[id] || "red";
      return { id, dx, dy, state };
    });

    // Arrow pointing toward intersection
    const arrowDist = intW + armLen / 2;
    const ax = cx + Math.cos(rad) * arrowDist;
    const ay = cy + Math.sin(rad) * arrowDist;
    const arrowRot = lane.angle + 180;

    return (
      <g>
        {/* Road surface */}
        <rect
          x={rx} y={ry} width={rw} height={rh}
          fill="var(--bg-raised)"
          stroke="var(--border)"
          strokeWidth={0.5}
        />
        {/* Belief heat tint */}
        <rect
          x={rx} y={ry} width={rw} height={rh}
          fill={fillColor}
          style={{ transition: "fill 400ms ease" }}
        />
        {/* Centre dashed line along road */}
        <line
          x1={cx + Math.cos(rad) * intW}
          y1={cy + Math.sin(rad) * intW}
          x2={cx + Math.cos(rad) * (intW + armLen)}
          y2={cy + Math.sin(rad) * (intW + armLen)}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={1}
          strokeDasharray="6 6"
        />
        {/* Direction arrow */}
        <text
          x={ax} y={ay}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={12}
          fill={active ? "var(--green)" : "var(--text-dim)"}
          transform={`rotate(${arrowRot}, ${ax}, ${ay})`}
          style={{ pointerEvents: "none", transition: "fill 300ms ease" }}
        >
          →
        </text>

        {/* TLS dots */}
        {tlsDots.map(({ id, dx, dy, state }) => (
          <g key={id}>
            <circle cx={dx} cy={dy} r={6} fill="var(--bg-base)" stroke="var(--border)" strokeWidth={1} />
            <circle
              cx={dx}
              cy={dy}
              r={4}
              fill={tlsColor(state)}
              style={{ transition: "fill 300ms ease", filter: state === "green" ? "drop-shadow(0 0 4px rgba(0,230,118,0.8))" : "none" }}
            />
          </g>
        ))}

        {/* Lane label */}
        <text
          x={cx + Math.cos(rad) * (intW + armLen + 10)}
          y={cy + Math.sin(rad) * (intW + armLen + 10)}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={11}
          fontWeight={700}
          fontFamily="var(--font-mono)"
          fill={active ? "var(--green)" : phase === "ARMED" ? "var(--amber)" : "var(--text-muted)"}
          style={{ transition: "fill 300ms ease" }}
        >
          {lane.label}
        </text>
      </g>
    );
  }

  // Preempt flash ring around intersection when any lane fires
  const anyActive = activeLanes.length > 0;

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div className="card-title" style={{ alignSelf: "flex-start" }}>
        <span className="dot-accent" style={{ background: anyActive ? "var(--green)" : "var(--text-dim)" }} />
        Intersection View
        {anyActive && (
          <span className="upper" style={{ color: "var(--green)", marginLeft: 8, fontSize: 10 }}>
            ⚡ PREEMPTING
          </span>
        )}
      </div>

      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Roads */}
        {lanes.map(l => <Road key={l.id} lane={l} />)}

        {/* Intersection box */}
        <rect
          x={cx - intW}
          y={cy - intW}
          width={intW * 2}
          height={intW * 2}
          fill="var(--bg-raised)"
          stroke="var(--border)"
          strokeWidth={1}
        />

        {/* Active preemption glow ring */}
        {anyActive && (
          <circle
            cx={cx}
            cy={cy}
            r={intW + 4}
            fill="none"
            stroke="var(--green)"
            strokeWidth={1.5}
            opacity={0.6}
            style={{ animation: "preempt-ring 1.2s ease-in-out infinite" }}
          />
        )}

        {/* Intersection centre logo */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          fontSize={9}
          fontWeight={700}
          fontFamily="var(--font-mono)"
          fill="var(--text-dim)"
        >
          GW++
        </text>
        <text
          x={cx}
          y={cy + 8}
          textAnchor="middle"
          fontSize={8}
          fontFamily="var(--font-mono)"
          fill="var(--text-dim)"
        >
          {new Date().toLocaleTimeString()}
        </text>
      </svg>

      <style>{`
        @keyframes preempt-ring {
          0%, 100% { opacity: 0.4; r: ${intW + 4}; }
          50%       { opacity: 0.9; r: ${intW + 12}; }
        }
      `}</style>
    </div>
  );
}
