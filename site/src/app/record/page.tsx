import data from "@/lib/final-results.json";
import { SwapDiagram } from "@/components/swap-diagram";

const repo = "https://github.com/basavarajnaduvinamani/DA5410-Winter-Project";
const record = (path: string) => `${repo}/blob/v2.0-final/${path}`;
const signed = (n: number) => `${n < 0 ? "−" : "+"}${Math.abs(n).toFixed(3)}`;
const interval = (v: number[]) => `[${v.map(signed).join(", ")}]`;
const percentage = (n: number) => (100 * n).toFixed(1);
const { gemma, qwen, exploratory: e } = data;
const labels = ["Sound", "Video"];
const chapters = ["Seeing things", "Does it listen?", "A second model disagrees", "Fixes I tried", "Rules set before looking", "What I’d test next"];

function Heading({ index }: { index: number }) {
  return <div className="section-heading"><span className="section-index">0{index + 1}</span><h2>{chapters[index]}</h2></div>;
}

function Forest() {
  const rows = [...gemma, ...qwen];
  const names = ["Gemma sound", "Gemma video", "Qwen sound", "Qwen video"];
  const x = (v: number) => 160 + v * 400;
  return <figure>
    <svg viewBox="0 0 320 192" style={{ overflow: "visible" }} role="img" aria-label="Forest plot of paired AUROC differences. Only the Qwen sound interval excludes zero.">
      <path d="M160 6V145" stroke="var(--muted)" strokeDasharray="3 3" />
      {rows.map((r, i) => {
        const y = 19 + i * 32;
        const color = r.falseYes < 40 ? "var(--muted)" : "var(--accent)";
        return <g key={names[i]} fill={color} stroke={color}>
          <text x="2" y={y - 7} stroke="none" fontSize="10">{names[i]}{r.falseYes < 40 ? ", underpowered" : ""}</text>
          <path d={`M${x(r.ci[0])} ${y}H${x(r.ci[1])} M${x(r.ci[0])} ${y-3}v6 M${x(r.ci[1])} ${y-3}v6`} />
          <circle cx={x(r.delta)} cy={y} r="3" />
        </g>;
      })}
      {[-.25, 0, .25].map(v => <text key={v} x={x(v)} y="163" textAnchor="middle" fontSize="10" fill="currentColor">{v.toFixed(2)}</text>)}
      <text x="160" y="184" textAnchor="middle" fontSize="11" fill="currentColor">ΔAUROC (probe minus baseline)</text>
    </svg>
    <figcaption className="caption">Only Qwen sound clears zero (held-out, n = Yes answers).</figcaption>
    <ul className="plain-list mono">{rows.map((r,i) => <li key={names[i]}>{names[i]} {signed(r.delta)} {interval(r.ci)}, n={r.yes}, {r.falseYes} false Yes</li>)}</ul>
  </figure>;
}

function Tradeoff() {
  const points = qwen.flatMap((task, i) => task.filters.map(f => ({ ...f, task: labels[i] })));
  const x = (v: number) => 36 + v * 1780;
  const y = (v: number) => 128 - v * 470;
  return <figure>
    <svg viewBox="0 0 320 176" role="img" aria-label="Qwen policy trade-off. The sound probe is beyond the 2 percentage point recall cap. The video probe falls within the cap and is underpowered.">
      <rect x={x(.02)} y="12" width={285-x(.02)} height="116" fill="var(--soft)" />
      <path d={`M36 12V128H285 M${x(.02)} 12V128`} stroke="var(--muted)" fill="none" strokeDasharray="3 2" />
      {points.map((p, i) => <g key={i} stroke={p.task === "Sound" ? "var(--accent)" : "var(--muted)"}>
        <path d={`M${x(p.cost.estimate)} ${y(p.cut.ci95[0])}V${y(p.cut.ci95[1])}`} />
        <circle cx={x(p.cost.estimate)} cy={y(p.cut.estimate)} r="6" fill="var(--surface)" />
        <text x={x(p.cost.estimate)} y={y(p.cut.estimate) + 3} textAnchor="middle" fontSize="8" fill="currentColor" stroke="none">{i + 1}</text>
      </g>)}
      <g fontSize="10" fill="currentColor">
        <text x="36" y="10">False-Yes reduction (pp)</text>
        <text x="9" y={y(.2)+3}>20</text><text x="9" y={y(.1)+3}>10</text><text x="14" y="131">0</text>
        <text x={x(.02)+4} y="142">2-pt cap</text>
        <text x="36" y="142">0</text><text x={x(.1)} y="142">10</text>
        <text x="155" y="166" textAnchor="middle">Recall drop (percentage points)</text>
      </g>
    </svg>
    <figcaption className="caption">The sound probe exceeded the recall cap (held-out, sound n={qwen[0].n}, video n={qwen[1].n}). Numbered points match the list below; lines show 95% CIs for false-Yes reduction.</figcaption>
    <ul className="plain-list mono">{points.map((p, i) => <li key={p.task+p.name}>{i + 1}. {p.task} {p.name.toLowerCase()}: {percentage(p.cut.estimate)} pp cut, {percentage(p.cost.estimate)} pp recall drop</li>)}</ul>
  </figure>;
}

function MirroredRates() {
  return <figure>
    <svg viewBox="0 0 320 145" role="img" aria-label="Mirrored Yes-rate and recall bars. Gemma sound recall is 10.1 percent and Qwen sound recall is 92.7 percent.">
      <path d="M159 23V129" stroke="var(--line)" />
      <g fill="currentColor" fontSize="10">
        <text x="8" y="15">Gemma 4 E2B</text><text x="235" y="15">Qwen2.5-Omni</text>
        {[0,1,2,3].map(i => {
          const task = Math.floor(i/2), key = i%2 ? "recall" : "yesRate";
          const a = gemma[task][key], b = qwen[task][key], y = 36 + i*29;
          return <g key={i}><text x="160" y={y-5} textAnchor="middle">{labels[task]} {i%2 ? "recall" : "Yes rate"}</text>
            <rect x={159-130*a} y={y} width={130*a} height="8" fill="var(--muted)" />
            <rect x="160" y={y} width={130*b} height="8" fill="var(--accent)" />
            <text x={151-130*a} y={y+8} textAnchor="end">{percentage(a)}%</text>
            <text x={165+130*b} y={y+8}>{percentage(b)}%</text></g>;
        })}
      </g>
    </svg>
    <figcaption className="caption">Sound recall was lower than video recall for Gemma and higher for Qwen (held-out, task n in the forest plot’s underlying records).</figcaption>
  </figure>;
}

export default function Home() {
  return <div className="research-layout">
    <aside className="page-toc" aria-label="On this page"><p className="eyebrow">Research record</p><nav>
      {chapters.map((chapter,i) => <a key={chapter} href={`#chapter-${i+1}`}>{chapter}</a>)}
    </nav></aside>
    <article className="research-article">
      <header className="hero">
        <p className="eyebrow">CounterPrior, by Basavaraj A Naduvinamani</p>
        <p className="caption">I take small multimodal models apart by muting, swapping and probing what they’re given, to find out what they actually perceive, and I report what didn’t work as carefully as what did.</p>
        <h1>Small on-device AI models don’t see and hear equally.</h1>
        <p className="lead">I measured where they make things up across image, audio and video, and chose not to ship my own detector.</p>
        <p>A hard-of-hearing user asks a phone assistant, “Is that a siren?” On held-out sound questions, Gemma 4 E2B caught only {percentage(gemma[0].recall)}% of the sounds that were really there.</p>
        <div className="tldr"><p>On held-out Qwen2.5-Omni-3B sound questions, a hidden-state probe improved false-Yes detection by ΔAUROC* {signed(qwen[0].delta)} (95% CI* {interval(qwen[0].ci)}), test run once. A probe filter cut false Yes by 12.4 points at a 3.2-point recall cost, above our preset 2-point cap, so it did not ship.</p></div>
      </header>
      <section id="chapter-1" className="paper-section"><Heading index={0} />
        <p className="eyebrow">LLaVA-1.5 at AI4Bharat, November 2025 to February 2026</p>
        <p>I designed CounterPrior-50 during the internship. The September 2026 release has 50 images, with 18 attribute traps, 18 absence traps and 14 positive controls.</p>
        <p>A later verified POPE* run scored 1,800 held-out questions with accuracy 0.863 [0.840, 0.888]. The hidden-state probe reached 0.719 AUROC versus 0.745 for answer confidence. Steering did not beat the matched controls.</p>
        <p className="endpoint-note">The probe lost, so I kept going.</p>
        <a href={record("results/REPORT_metrics.json")}>POPE results</a>
      </section>
      <section id="chapter-2" className="paper-section"><Heading index={1} />
        <p className="eyebrow">Gemma 4 E2B, follow-up, November 2026</p>
        <p>Silencing audio shifted the Yes–No score by {e.silence.estimate.toFixed(2)} on average and flipped {e.silenceFlips} of {e.silence.n} answers.</p>
        <SwapDiagram />
        <p className="caption"><em>Exploratory.</em> Video swaps moved sound-question scores {e.soundSwap.estimate.toFixed(2)}× as much as audio swaps [{e.soundSwap.ci95.map(v => v.toFixed(2)).join(", ")}] (calibration and validation, n={e.soundSwap.n}).</p>
        <p className="endpoint-note">Gemma had 15 sound and 24 video false-Yes cases, below the 40-case threshold for each task.</p>
        <a href={record("results/v2/gemma-exploratory-summary.json")}>Gemma exploratory results</a>
      </section>
      <section id="chapter-3" className="paper-section"><Heading index={2} />
        <p className="eyebrow">Qwen2.5-Omni-3B, follow-up, November 2026</p>
        <p>Qwen’s held-out raw-margin AUROC was {qwen[0].auc.estimate.toFixed(3)} on sound questions and {qwen[1].auc.estimate.toFixed(3)} on video questions.</p>
        <MirroredRates />
        <Forest />
      </section>
      <section id="chapter-4" className="paper-section"><Heading index={3} />
        <p>Gemma audio-only gained {signed(e.audioOnly.estimate)} {interval(e.audioOnly.ci95)} and Listen First gained {signed(e.listenFirst.estimate)} {interval(e.listenFirst.ci95)} on validation (n={e.audioOnly.n}). Both intervals included zero. The contrast scores trailed the raw baseline.</p>
        <Tradeoff />
        <p className="endpoint-note">I held back the Qwen sound filter because of its recall cost. The video probe met the demo rule with only 26 false-Yes cases, below the power threshold.</p>
      </section>
      <section id="chapter-5" className="paper-section"><Heading index={4} />
        <p>Calibration fit the probes. Validation chose layers and thresholds. Each held-out evaluation ran once after choices were frozen. Each model used a 100-item label audit. The Kaggle setup used three accounts and six T4 GPUs for the Gemma benchmark.</p>
        <ul className="plain-list"><li>Early pilot figures were withdrawn after a scorer audit.</li><li>The 3.6% flip claim was withdrawn as evidence of sound removal after the donor check found that most donor clips lacked a verified absence label.</li><li>Released clip IDs stand in for source lineage where original sources are unknown. The Gemma swap findings are exploratory.</li></ul>
      </section>
      <section id="chapter-6" className="paper-section"><Heading index={5} />
        <ul className="plain-list"><li>Targeted swaps with independently verified donor labels.</li><li>Attention to audio tokens during sound questions.</li><li>Fine-tuning with audio-swap negatives.</li></ul>
        <p>AVHBench* access and the AVCD* baseline are described in the preregistration.</p>
        <div className="resource-links"><a href={`${repo}/tree/v2.0-final`}>Repository</a><a href={record("docs/PREREGISTRATION_KAGGLE.md")}>Preregistration</a></div>
        <p className="caption">* AUROC, area under the receiver operating characteristic curve. CI, confidence interval. POPE, Polling-based Object Probing Evaluation. AVHBench, Audio-Visual Hallucination Benchmark. AVCD, Audio-Visual Contrastive Decoding.</p>
      </section>
    </article>
  </div>;
}
