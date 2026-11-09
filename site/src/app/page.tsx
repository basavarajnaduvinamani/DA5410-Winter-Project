import { SavedExample } from "@/components/saved-example";
import { code, research } from "@/lib/data";
import results from "@/lib/final-results.json";

function whisker(left: number, right: number, y: number) {
  return `M${left} ${y}H${right} M${left} ${y - 5}v10 M${right} ${y - 5}v10`;
}

function LLaVAChart() {
  const detectors = research.results.detection?.detectors;
  const metrics = [
    { label: "Probe", score: detectors?.probe.auroc, color: "var(--accent)" },
    { label: "Model confidence", score: detectors?.neg_logprob.auroc, color: "var(--muted)" },
  ];
  const x = (v: number) => 16 + 328 * v;

  return (
    <figure className="result-plot">
      <h3>LLaVA on POPE</h3>
      <p>1,800 questions. Answer accuracy: 0.863.</p>
      <svg viewBox="0 0 360 178" role="img" aria-label="Probe AUROC 0.719. Model confidence AUROC 0.745. Whiskers show saved 95% confidence intervals.">
        {metrics.map(({ label, score, color }, i) => {
          if (score?.estimate == null) return null;
          const y = 41 + 54 * i;
          return <g key={label}>
            <text x={x(0)} y={y - 12}>{label}</text>
            <text x={x(1)} y={y - 12} textAnchor="end">{score.estimate.toFixed(3)}</text>
            <rect x={x(0)} y={y} width={x(score.estimate) - x(0)} height="8" rx="4" fill={color} />
            {score.ci95 && <path d={whisker(x(score.ci95[0]), x(score.ci95[1]), y + 4)} className="bar-whisker" />}
          </g>;
        })}
        <path d={`M${x(0)} 127H${x(1)}`} className="plot-axis" />
        {[0, 0.5, 1].map(v => <text key={v} x={x(v)} y="147" textAnchor="middle">{v}</text>)}
        <text x="180" y="170" textAnchor="middle">AUROC</text>
      </svg>
      <figcaption>The probe did not beat model confidence. Bars: AUROC; whiskers: 95% bootstrap CI.</figcaption>
    </figure>
  );
}

function QwenChart() {
  const { delta, ci } = results.qwen[0];
  const x = (v: number) => 24 + 780 * (v + 0.1);
  const signed = (v: number) => `+${v.toFixed(3)}`;
  const summary = `${signed(delta)} (95% CI ${signed(ci[0])} to ${signed(ci[1])})`;
  return (
    <figure className="result-plot">
      <h3>Qwen2.5-Omni-3B audio</h3>
      <p>{summary}</p>
      <svg viewBox="0 0 360 178" role="img" aria-label={`AUROC difference: ${summary}`}>
        <text x={x(0)} y="30" textAnchor="middle">No difference</text>
        <path d={`M${x(0)} 40V127`} className="plot-axis zero-line" />
        <path d={`M${x(-0.1)} 127H${x(0.3)}`} className="plot-axis" />
        <path d={whisker(x(ci[0]), x(ci[1]), 79)} className="difference-whisker" />
        <circle cx={x(delta)} cy="79" r="5" fill="var(--accent)" />
        {[-0.1, 0, 0.1, 0.2, 0.3].map(v => <text key={v} x={x(v)} y="147" textAnchor="middle">{v > 0 ? `+${v}` : v}</text>)}
        <text x="180" y="170" textAnchor="middle">AUROC difference</text>
      </svg>
      <figcaption>Dot: AUROC difference; whiskers: 95% bootstrap CI. The baseline uses Qwen’s own Yes–No scores: the original score, its change after audio swaps, the score without media, and how often Qwen said Yes for that object and task in calibration.</figcaption>
    </figure>
  );
}

export default function Home() {
  const example = research.examples.find(row => row.baselinePred !== row.steeredPred);
  return (
    <article className="redesign">
      <header>
        <div className="question-copy">
          <h1>Do multimodal models know when they are guessing?</h1>
          <p>How do machines see? A workbench for tinkering with open-weight multimodal models.</p>
          <p>We tested whether a probe of a model’s hidden states can spot wrong answers better than the model’s own scores.</p>
          <p className="research-credit">LLaVA work at AI4Bharat, IIT Madras (Prof. Mitesh Khapra), Nov 2025 to Feb 2026. Extended to audio and video in Nov 2026.</p>
        </div>
      </header>

      <section id="results" className="redesign-section opening-results">
        <h2>Results</h2>
        <p>AUROC measures how well a score ranks errors above correct answers. Higher is better.</p>
        <div className="result-grid"><LLaVAChart /><QwenChart /></div>
        <h3 className="swap-heading">Gemma 4 E2B, audio and video</h3>
        <p className="swap-finding">Inconclusive: too few wrong answers to compare. Exploratory only: video swaps shifted sound-question scores {results.exploratory.soundSwap.estimate.toFixed(2)}× as much as audio swaps.</p>
        <a href="/record/">Complete results and swap diagram</a>
      </section>

      <div className="reading-column">
        {example && <SavedExample example={example} />}
        <section id="method" className="redesign-section">
          <h2>Method</h2>
          <p>We fit layer-wise linear probes on hidden states. The image experiment used image-disjoint splits. We selected settings before held-out evaluation and used bootstrap confidence intervals.</p>
        </section>
        <section id="limitations" className="redesign-section">
          <h2>Limitations</h2>
          <p>The swap finding is exploratory. A score shift alone cannot establish which stream the model understood. The interactive example shows a saved answer change and does not run the model.</p>
        </section>
        <section id="code" className="redesign-section">
          <h2>Code and evidence</h2>
          <div className="redesign-links"><a href={code}>GitHub repository</a><a href="/explorer/">Example explorer</a><a href="/ablations/">Ablations</a></div>
        </section>
      </div>
    </article>
  );
}
