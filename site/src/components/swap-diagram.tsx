"use client";

import { useState } from "react";

const modes = ["Full", "Swap audio", "Swap video", "Silence", "No media"] as const;
const inputs = [
  ["Original video", "Original audio"],
  ["Original video", "Donor audio"],
  ["Donor video", "Original audio"],
  ["Original video", "Silenced audio"],
  ["No video", "No audio"],
];

export function SwapDiagram() {
  const [mode, setMode] = useState(0);
  return <div className="evidence-note swap-demo">
    <div className="swap-controls" aria-label="Input intervention">
      {modes.map((label, i) => <button key={label} type="button" aria-pressed={mode === i}
        onClick={() => setMode(i)}>{label}</button>)}
    </div>
    <svg viewBox="0 0 320 105" role="img" aria-label={inputs[mode].join(" and ") + " feed the model"}>
      <g fill="var(--surface)" stroke="var(--line)">
        <rect x="1" y="3" width="125" height="36" /><rect x="1" y="65" width="125" height="36" />
        <rect x="196" y="34" width="123" height="36" />
      </g>
      <path d="M126 21H160V52H195 M126 83H160V52 M188 47L195 52L188 57" fill="none" stroke="var(--accent)" />
      <g fill="currentColor" textAnchor="middle" fontSize="13">
        <text x="63" y="26">{inputs[mode][0]}</text><text x="63" y="88">{inputs[mode][1]}</text>
        <text x="258" y="57">Yes / No score</text>
      </g>
    </svg>
  </div>;
}
