import { research, metric, profileLabel } from "@/lib/data";
import { Source } from "./shell";
import { methods } from "./results";
export function LayerChart() {
  const points = research.calibrationCurve;
  if (!points.length) return null;
  const min = Math.min(...points.map((p) => p.layer)),
    max = Math.max(...points.map((p) => p.layer));
  const x = (l: number) => 64 + ((l - min) / Math.max(1, max - min)) * 650;
  const y = (v: number) => 260 - v * 210;
  const curve = points.filter((p) => p.probe !== null);
  return (
    <figure className="data-figure">
      <h3>Where does error information become readable?</h3>
      <p className="profile-caption">{profileLabel()}</p>
      <div
        className="chart-scroll"
        role="region"
        tabIndex={0}
        aria-label="Layer sweep chart, horizontally scrollable"
      >
        <svg viewBox="0 0 760 315" role="img" aria-labelledby="layer-title">
          <title id="layer-title">
            Calibration cross-validated probe AUROC by decoder layer. Selection
            diagnostic; no calibration confidence intervals were stored.
          </title>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <path
                d={`M64 ${y(t)}H714`}
                stroke="var(--line)"
                strokeDasharray={t === 0.5 ? "4 4" : undefined}
              />
              <text
                x="49"
                y={y(t) + 4}
                textAnchor="end"
                fill="var(--muted)"
                fontSize="12"
              >
                {t}
              </text>
            </g>
          ))}
          <path
            d={curve
              .map((p, i) => `${i ? "L" : "M"}${x(p.layer)} ${y(p.probe!)}`)
              .join(" ")}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2.5"
          />
          {curve.map((p) => (
            <circle
              key={p.layer}
              cx={x(p.layer)}
              cy={y(p.probe!)}
              r="3"
              fill="var(--accent)"
            />
          ))}
          {points
            .filter((_, i) => i === 0 || i === points.length - 1 || i % 5 === 0)
            .map((p) => (
              <text
                key={p.layer}
                x={x(p.layer)}
                y="282"
                fill="var(--muted)"
                textAnchor="middle"
                fontSize="12"
              >
                {p.layer}
              </text>
            ))}
          <text
            x="390"
            y="308"
            textAnchor="middle"
            fill="var(--ink)"
            fontSize="13"
          >
            Decoder layer · zero-based raw block output
          </text>
          <text
            x="18"
            y="155"
            transform="rotate(-90 18 155)"
            textAnchor="middle"
            fill="var(--ink)"
            fontSize="13"
          >
            Calibration CV AUROC
          </text>
        </svg>
      </div>
      <figcaption>
        Layer choice uses calibration only. This diagnostic curve has no stored
        confidence intervals; it is not the held-out result. The test table
        reports image-bootstrap intervals.
      </figcaption>
      <Source path="results/REPORT_metrics.json" />
    </figure>
  );
}
export function FalseYesChart({
  headingLevel = 3,
}: { headingLevel?: 2 | 3 } = {}) {
  if (!research.run) return null;
  const all = [["baseline", "Baseline"], ...methods];
  const rows = ["random", "popular", "adversarial"]
    .flatMap((split) =>
      all.map(([key, label]) => ({
        split,
        key,
        label,
        m:
          key === "baseline"
            ? research.results.baseline[`test:${split}`]?.false_yes_rate
            : research.results.steering[key]?.per_split_metrics?.[split]
                ?.false_yes_rate,
      })),
    )
    .filter((r) => r.m?.ci95 && r.m.estimate !== null);
  if (!rows.length) return null;
  const upper = Math.max(
    0.2,
    Math.min(1, Math.ceil(Math.max(...rows.map((r) => r.m!.ci95![1])) * 5) / 5),
  );
  const x = (v: number) => 226 + (v / upper) * 480;
  const Heading = headingLevel === 2 ? "h2" : "h3";
  return (
    <figure className="data-figure">
      <Heading>False-yes rate, including both controls</Heading>
      <p className="profile-caption">{profileLabel()}</p>
      <div
        className="chart-scroll"
        role="region"
        tabIndex={0}
        aria-label="False-yes rates chart, horizontally scrollable"
      >
        <svg
          viewBox={`0 0 760 ${rows.length * 32 + 74}`}
          role="img"
          aria-labelledby="bars-title"
        >
          <title id="bars-title">
            False-yes rates by POPE split and method, with image-bootstrap
            confidence whiskers. Lower is better.
          </title>
          {Array.from({ length: 5 }, (_, i) => (upper * i) / 4).map((t) => (
            <g key={t}>
              <path
                d={`M${x(t)} 16V${rows.length * 32 + 22}`}
                stroke="var(--line)"
              />
              <text
                x={x(t)}
                y={rows.length * 32 + 44}
                textAnchor="middle"
                fill="var(--muted)"
                fontSize="12"
              >
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {rows.map((r, i) => {
            const yy = 24 + i * 32;
            const ci = r.m!.ci95!;
            return (
              <g key={r.key + r.split}>
                <text
                  x="212"
                  y={yy + 4}
                  textAnchor="end"
                  fill="var(--ink)"
                  fontSize="11"
                >
                  {r.split} · {r.label}
                </text>
                <rect
                  x={x(0)}
                  y={yy - 6}
                  width={Math.max(1, x(r.m!.estimate!) - x(0))}
                  height="12"
                  fill={
                    r.key === "slerp_pca"
                      ? "var(--accent)"
                      : "var(--chart-muted)"
                  }
                  opacity={r.key === "baseline" ? 1 : 0.8}
                />
                <path
                  d={`M${x(ci[0])} ${yy}H${x(ci[1])}M${x(ci[0])} ${yy - 4}v8M${x(ci[1])} ${yy - 4}v8`}
                  stroke="var(--ink)"
                  strokeWidth="1.5"
                />
                <title>{`${r.label}, ${r.split}: ${metric(r.m)}`}</title>
              </g>
            );
          })}
          <text
            x="470"
            y={rows.length * 32 + 69}
            textAnchor="middle"
            fill="var(--ink)"
            fontSize="13"
          >
            False-yes rate on absent objects · lower is better
          </text>
        </svg>
      </div>
      <figcaption>
        Whiskers show stored image-bootstrap confidence intervals. No bar is
        drawn when its interval is unavailable. The axis starts at zero and
        extends beyond every interval. Scroll horizontally on narrow screens to
        read every label.
      </figcaption>
      <Source path="results/REPORT_metrics.json" />
    </figure>
  );
}
