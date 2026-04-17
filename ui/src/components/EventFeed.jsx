import React, { useEffect, useRef, useState } from "react";

/**
 * Event feed — scrollable log of preemption events and system state changes.
 *
 * Props:
 *   events  [{ id, timestamp, type, lane, belief, etas, message }]
 */
export default function EventFeed({ events = [] }) {
  const bottomRef = useRef(null);

  // Auto-scroll to latest event
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  function fmtTime(ts) {
    return new Date(ts * 1000).toLocaleTimeString("en-US", {
      hour12: false,
      hour:   "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function eventStyle(type) {
    if (type === "preempt")    return { color: "var(--green)", icon: "⚡" };
    if (type === "armed")      return { color: "var(--amber)", icon: "🔔" };
    if (type === "cooling")    return { color: "var(--blue)",  icon: "↓" };
    if (type === "connected")  return { color: "var(--green)", icon: "◉" };
    if (type === "disconnected") return { color: "var(--red)", icon: "◎" };
    return { color: "var(--text-muted)", icon: "·" };
  }

  return (
    <div
      className="card"
      style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, overflow: "hidden" }}
    >
      <div className="card-title">
        <span className="dot-accent" />
        Event Log
        <span
          className="mono"
          style={{ marginLeft: "auto", color: "var(--text-dim)", fontSize: 10 }}
        >
          {events.length} events
        </span>
      </div>

      <div
        style={{
          flex:       1,
          overflowY:  "auto",
          display:    "flex",
          flexDirection: "column",
          gap: 6,
          paddingRight: 4,
        }}
      >
        {events.length === 0 && (
          <div
            className="muted"
            style={{ textAlign: "center", paddingTop: 40, fontSize: 12 }}
          >
            Waiting for events…
          </div>
        )}

        {events.map(evt => {
          const { color, icon } = eventStyle(evt.type);
          return (
            <div
              key={evt.id}
              style={{
                background:    "var(--bg-raised)",
                border:        `1px solid ${color}28`,
                borderLeft:    `3px solid ${color}`,
                borderRadius:  "var(--r-sm)",
                padding:       "8px 10px",
                animation:     "fade-in 250ms ease",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <span style={{ color, fontWeight: 700, fontSize: 12 }}>
                  {icon} {evt.type.toUpperCase()}
                </span>
                <span className="mono muted" style={{ fontSize: 10 }}>
                  {fmtTime(evt.timestamp)}
                </span>
              </div>

              {evt.lane && (
                <div className="mono" style={{ fontSize: 11, color: "var(--text)", marginBottom: 2 }}>
                  Lane: <span style={{ color }}>{evt.lane.replace("approach_", "").toUpperCase()}</span>
                  {evt.belief !== undefined && (
                    <span className="muted" style={{ marginLeft: 8 }}>
                      belief={evt.belief.toFixed(3)}
                    </span>
                  )}
                </div>
              )}

              {evt.etas && evt.etas.length > 0 && (
                <div className="mono muted" style={{ fontSize: 10 }}>
                  ETAs: {evt.etas.map(e => `${e}s`).join(" → ")}
                </div>
              )}

              {evt.message && (
                <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>
                  {evt.message}
                </div>
              )}
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>

      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
