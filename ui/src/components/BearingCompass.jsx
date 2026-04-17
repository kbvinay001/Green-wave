import React, { useCallback, useEffect, useRef, useState } from "react";

/**
 * Bearing compass — shows the estimated audio bearing of the siren source.
 *
 * Props:
 *   bearing   (number)   azimuth in degrees, 0=N, 90=E, 180=S, 270=W
 *   confidence (number)  0–1, drives arc thickness + needle opacity
 *   detected   (bool)    whether siren is currently classified as active
 */
export default function BearingCompass({ bearing = 0, confidence = 0, detected = false }) {
  const size      = 220;
  const cx        = size / 2;
  const cy        = size / 2;
  const outerR    = 96;
  const innerR    = 72;
  const needleLen = 62;
  const needleW   = 2.5;

  // Smooth bearing changes — prevent needle jumping over 180° discontinuity
  const prevBearing = useRef(bearing);
  const [smoothBearing, setSmoothBearing] = useState(bearing);

  useEffect(() => {
    let diff = bearing - prevBearing.current;
    if (diff > 180)  diff -= 360;
    if (diff < -180) diff += 360;
    prevBearing.current = bearing;
    setSmoothBearing(b => b + diff);
  }, [bearing]);

  // Convert azimuth to SVG rotation: 0° is up (north), grows clockwise
  const rotation = smoothBearing - 90;   // SVG 0° is east; north needs –90

  // Confidence arc (partial ring around compass)
  const arcAngle  = Math.max(5, confidence * 120);   // degrees swept
  const arcStart  = (smoothBearing - arcAngle / 2 - 90) * (Math.PI / 180);
  const arcEnd    = (smoothBearing + arcAngle / 2 - 90) * (Math.PI / 180);
  const arcX1     = cx + outerR * Math.cos(arcStart);
  const arcY1     = cy + outerR * Math.sin(arcStart);
  const arcX2     = cx + outerR * Math.cos(arcEnd);
  const arcY2     = cy + outerR * Math.sin(arcEnd);
  const largeArc  = arcAngle > 180 ? 1 : 0;

  // Cardinal labels
  const cardinals = [
    { label: "N", angle: -90 },
    { label: "E", angle:   0 },
    { label: "S", angle:  90 },
    { label: "W", angle: 180 },
  ];

  const color = detected ? "var(--amber)" : "var(--text-muted)";

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div className="card-title" style={{ alignSelf: "flex-start" }}>
        <span className="dot-accent" style={{ background: detected ? "var(--amber)" : "var(--text-dim)" }} />
        Audio Bearing
      </div>

      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ filter: detected ? "drop-shadow(0 0 10px rgba(255,171,64,0.35))" : "none", transition: "filter 400ms ease" }}
      >
        {/* Background rings */}
        <circle cx={cx} cy={cy} r={outerR} fill="none" stroke="var(--border)" strokeWidth={1} />
        <circle cx={cx} cy={cy} r={innerR} fill="none" stroke="var(--border)" strokeWidth={0.5} />
        <circle cx={cx} cy={cy} r={3}      fill="var(--text-dim)" />

        {/* Tick marks at every 30° */}
        {Array.from({ length: 12 }, (_, i) => {
          const a  = (i * 30 - 90) * (Math.PI / 180);
          const r1 = outerR - 4;
          const r2 = outerR - 10;
          return (
            <line
              key={i}
              x1={cx + r1 * Math.cos(a)} y1={cy + r1 * Math.sin(a)}
              x2={cx + r2 * Math.cos(a)} y2={cy + r2 * Math.sin(a)}
              stroke="var(--text-dim)"
              strokeWidth={i % 3 === 0 ? 1.5 : 0.7}
            />
          );
        })}

        {/* Cardinal labels */}
        {cardinals.map(({ label, angle }) => {
          const a = angle * (Math.PI / 180);
          const r = outerR - 18;
          return (
            <text
              key={label}
              x={cx + r * Math.cos(a)}
              y={cy + r * Math.sin(a)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={10}
              fontFamily="var(--font-mono)"
              fontWeight={700}
              fill={label === "N" ? "var(--green)" : "var(--text-muted)"}
            >
              {label}
            </text>
          );
        })}

        {/* Confidence arc */}
        {confidence > 0.05 && (
          <path
            d={`M ${arcX1} ${arcY1} A ${outerR} ${outerR} 0 ${largeArc} 1 ${arcX2} ${arcY2}`}
            fill="none"
            stroke={color}
            strokeWidth={3}
            strokeLinecap="round"
            opacity={0.7 + confidence * 0.3}
          />
        )}

        {/* Needle */}
        <g transform={`rotate(${rotation}, ${cx}, ${cy})`}>
          {/* Main needle */}
          <line
            x1={cx}
            y1={cy}
            x2={cx + needleLen}
            y2={cy}
            stroke={color}
            strokeWidth={needleW}
            strokeLinecap="round"
            opacity={0.2 + confidence * 0.8}
          />
          {/* Back stub */}
          <line
            x1={cx}
            y1={cy}
            x2={cx - 18}
            y2={cy}
            stroke="var(--text-dim)"
            strokeWidth={needleW * 0.6}
            strokeLinecap="round"
          />
          {/* Tip dot */}
          <circle
            cx={cx + needleLen}
            cy={cy}
            r={3}
            fill={color}
            opacity={0.2 + confidence * 0.8}
          />
        </g>

        {/* Centre hub */}
        <circle cx={cx} cy={cy} r={5} fill="var(--bg-raised)" stroke={color} strokeWidth={1.5} />
      </svg>

      {/* Readout */}
      <div style={{ textAlign: "center" }}>
        <div
          className="mono"
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: detected ? "var(--amber)" : "var(--text)",
            transition: "color 300ms ease",
            letterSpacing: "0.04em",
          }}
        >
          {Math.round(((bearing % 360) + 360) % 360).toString().padStart(3, "0")}°
        </div>
        <div className="muted upper" style={{ fontSize: 10, marginTop: 2 }}>
          conf {(confidence * 100).toFixed(0)}%
          {detected && <span style={{ color: "var(--amber)", marginLeft: 8 }}>● SIREN</span>}
        </div>
      </div>
    </div>
  );
}
