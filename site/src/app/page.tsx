import Link from "next/link";
import { research, code, colab, sourceUrl, profileLabel } from "@/lib/data";
import { PipelineFigure, GeometryFigure } from "@/components/method-figures";
import { ResultsOverview } from "@/components/results";
import { LayerChart, FalseYesChart } from "@/components/charts";
import { Source } from "@/components/shell";
const p = research.protocol;
export default function Home() {
  return (
    <div className="research-layout">
      <aside className="page-toc" aria-label="On this page">
        <p className="eyebrow">RESEARCH RECORD</p>
        <nav aria-label="Research sections">
          <a href="#question">Question</a>
          <a href="#method">Method</a>
          <a href="#protocol">Protocol</a>
          <a href="#results">Results</a>
          <a href="#corrections">What did not work</a>
          <a href="#limitations">Limitations</a>
          <a href="#reproduce">Reproduce</a>
        </nav>
        <span className="page-toc-foot">
          AI4Bharat Lab
          <br />
          IIT Madras
        </span>
      </aside>
      <article className="research-article">
        <header className="hero">
          <p className="eyebrow">
            MULTIMODAL REPRESENTATIONS / RESEARCH AT IIT MADRAS
          </p>
          <h1>
            CounterPrior: Detecting and Steering Object Hallucination from
            LLaVA&apos;s Hidden States
          </h1>
          <p className="author">
            Basavaraj A Naduvinamani <span>· AI4Bharat Lab, IIT Madras</span>
          </p>
          <p className="credits">
            Supervisor: Prof. Mitesh M. Khapra
            <br />
            Mentor: Mohammed Safi Ur Rahman Khan
          </p>
          <div className="resource-links">
            <a className="primary-link" href={code}>
              Code <span aria-hidden="true">↗</span>
            </a>
            <a href={colab}>
              Colab <span aria-hidden="true">↗</span>
            </a>
            <a href="/technical-report.pdf">Technical report PDF</a>
            <a href="/citation.bib" download>
              BibTeX
            </a>
          </div>
          <div className="tldr">
            <span className="eyebrow">THE CURRENT FINDING</span>
            {research.status === "pending" ? (
              <p>
                The protocol now separates calibration, validation and test
                images, but no pretrained-model result has been verified.
                Whether a hidden-state probe outperforms answer-token
                confidence—and whether SLERP reduces false-yes answers without
                recognition loss—remains inconclusive.
              </p>
            ) : (
              <p>{research.summary.join(" ")}</p>
            )}
            <Source
              path={
                research.status === "pending"
                  ? "results/2026-11-01/gpu_status.json"
                  : "results/REPORT_metrics.json"
              }
            />
          </div>
        </header>
        <section id="question" className="paper-section">
          <div className="section-heading">
            <span className="section-index">01</span>
            <h2>Can we see an error forming?</h2>
          </div>
          <p className="lead">
            When visual evidence and a familiar language prior disagree, a
            vision-language model can answer with the prior. CounterPrior asks
            whether that error is readable inside the decoder—and whether a
            small geometric intervention changes the answer.
          </p>
          <div className="question-pair">
            <div>
              <h3>Detection</h3>
              <p>
                Read the terminal prompt state before generation. Compare a
                linear probe with simple confidence baselines on images the
                probe never saw.
              </p>
            </div>
            <div>
              <h3>Intervention</h3>
              <p>
                Rotate an activation toward a calibration-learned direction.
                Measure absent-object false positives alongside recall on
                objects that are present.
              </p>
            </div>
          </div>
          <p className="endpoint-note">
            <strong>A distinction that matters:</strong> the active detector
            predicts <em>any wrong answer</em>, including false negatives.
            False-yes hallucination is a narrower steering outcome. Answer-token
            log-probability and entropy are later, first-token baselines—not
            early-warning probes.
          </p>
          <Source path="configs/pope_llava.json" />
        </section>
        <section id="method" className="paper-section">
          <div className="section-heading">
            <span className="section-index">02</span>
            <h2>Read the state. Test the direction.</h2>
          </div>
          <PipelineFigure />
          <div className="method-detail">
            <div>
              <h3>One readout, then a causal test</h3>
              <p>
                The probe reads the last prompt position, before a completion
                token can leak its wording into the representation. Each
                detector chooses its layer using image-grouped calibration
                folds.
              </p>
              <p>
                For steering, a separate cosine gate decides when to intervene.
                SLERP changes direction while preserving activation norm;
                mean-difference addition, random-direction and shuffled-label
                controls test competing explanations.
              </p>
              <p>
                A “truth axis” is an operational name for an oriented
                calibration direction. It is not a claim that truth is a
                universal feature.
              </p>
              <Source path="src/pope_pipeline.py" />
            </div>
            <GeometryFigure />
          </div>
        </section>
        <section id="protocol" className="paper-section">
          <div className="section-heading">
            <span className="section-index">03</span>
            <h2>Different images for each decision.</h2>
          </div>
          <p>
            POPE is the primary benchmark. All questions about the same COCO
            image stay in one partition, across random, popular and adversarial
            question sets.
          </p>
          <div
            className="table-scroll"
            role="region"
            aria-label="Frozen image partitions"
            tabIndex={0}
          >
            <table className="protocol-table">
              <caption>
                Frozen image partitions · planned protocol, not completed
                inference
              </caption>
              <thead>
                <tr>
                  <th scope="col">Partition</th>
                  <th scope="col">Full profile images</th>
                  <th scope="col">Quick profile images</th>
                  <th scope="col">What it decides</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["calibration", "Layer, probe, direction and gate"],
                  ["validation", "Steering strength"],
                  ["test", "Held-out reporting only"],
                ].map(([part, role]) => (
                  <tr key={part}>
                    <th scope="row">{part}</th>
                    <td>{p.counts[part]}</td>
                    <td>{p.quickCounts[part]}</td>
                    <td>{role}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Source path={p.source} />
          <dl className="protocol-details">
            <div>
              <dt>Scorer</dt>
              <dd>Official POPE yes/no parser, pinned question files</dd>
            </div>
            <div>
              <dt>Uncertainty</dt>
              <dd>
                {p.bootstrapIterations.toLocaleString("en-US")} paired
                image-cluster bootstrap resamples
              </dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd className="mono">{p.seed}</dd>
            </div>
            <div>
              <dt>Decision rule</dt>
              <dd>
                Intervals crossing the comparison boundary remain inconclusive
              </dd>
            </div>
          </dl>
          <Source path="configs/pope_llava.json" />
          <p className="caption">
            The bootstrap interval is conditional on frozen calibration and
            validation choices. The quick profile is exploratory. No settings
            are selected from test outcomes.
          </p>
        </section>
        <section id="results" className="paper-section">
          <div className="section-heading">
            <span className="section-index">04</span>
            <h2>Results, with the denominator visible.</h2>
          </div>
          <ResultsOverview />
          <LayerChart />
          <FalseYesChart />
          <div className="findings">
            {research.hypotheses.map((h) => (
              <div key={h.id}>
                <p className="eyebrow">
                  {h.id} <span className="finding-status">{h.status}</span>
                </p>
                <p>{h.finding}</p>
              </div>
            ))}
          </div>
          <div className="section-links">
            <Link href="/explorer/">
              Inspect examples <span aria-hidden="true">→</span>
            </Link>
            <Link href="/ablations/">
              Read every control <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
        <section id="corrections" className="paper-section">
          <div className="section-heading">
            <span className="section-index">05</span>
            <h2>What did not work—and what changed.</h2>
          </div>
          <p>
            The original custom benchmark was not reliable enough to carry the
            headline. Pixel review retained {p.audit.retained} of{" "}
            {p.audit.original} rows; the remaining {p.audit.excluded} had label
            or image-relevance problems. A scorer also accepted an unrelated
            negation as evidence that an absent object was not present.
          </p>
          <Source path="dataset/audit_2026-10-30.json" />
          <p>
            Earlier claims of perfect detection, a fixed hallucination reduction
            and preserved sightedness were withdrawn because complete supporting
            predictions and a valid held-out scoring record were not recovered.
            They are not reproduced as current results.
          </p>
          <p>
            The corrected custom set remains secondary. Proposed relabels have
            not been applied. POPE, matched controls and image-disjoint
            evaluation now carry the primary protocol.
          </p>
          <Source path="07_build_counterprior.md" />
        </section>
        <section id="limitations" className="paper-section">
          <div className="section-heading">
            <span className="section-index">06</span>
            <h2>What this study cannot establish.</h2>
          </div>
          <ul className="plain-list">
            <li>
              A POPE error signal is not a general hallucination detector;
              binary object questions do not cover long-form, temporal or audio
              grounding.
            </li>
            <li>
              Norm preservation does not establish a benefit. False-yes
              reduction must be read with recall loss and matched controls.
            </li>
            <li>
              Confidence intervals do not account for benchmark selection or
              every calibration uncertainty.
            </li>
            <li>
              The modern-model replication is registered but not GPU-verified.
              Architecture transfer remains an open question.
            </li>
            <li>
              No external user study, production safety impact or
              deployment-risk reduction has been measured.
            </li>
          </ul>
          <Source path="docs/PROTOCOL.md" />
        </section>
        <section id="reproduce" className="paper-section">
          <div className="section-heading">
            <span className="section-index">07</span>
            <h2>Reproduce the evidence.</h2>
          </div>
          <p>
            Start with the quick profile in Colab. The run directory stays on
            Drive; interrupted questions resume without choosing a new test
            configuration.
          </p>
          <div className="reproduce-actions">
            <a className="button-primary" href={colab}>
              Open the notebook <span aria-hidden="true">↗</span>
            </a>
            <a className="button-secondary" href="/research.json" download>
              Download evidence JSON
            </a>
          </div>
          <pre tabIndex={0} aria-label="Reproduction commands">
            <code>{`git clone --branch claude/site-2026-11 ${code.replace("/tree/claude/site-2026-11", "")}.git
cd DA5410-Winter-Project
bash scripts/colab_run.sh setup
COUNTERPRIOR_RUN=runs/quick COUNTERPRIOR_PROFILE=quick bash scripts/colab_run.sh timing
COUNTERPRIOR_RUN=runs/quick COUNTERPRIOR_PROFILE=quick bash scripts/colab_run.sh run
COUNTERPRIOR_RUN=runs/quick bash scripts/colab_run.sh report`}</code>
          </pre>
          <dl className="protocol-details">
            <div>
              <dt>Run identity</dt>
              <dd className="mono">
                {research.run?.id ?? "Not available — no verified run"}
              </dd>
            </div>
            <div>
              <dt>Reported profile</dt>
              <dd>{profileLabel()}</dd>
            </div>
            <div>
              <dt>Hardware</dt>
              <dd>
                {research.run?.hardware ?? "Not recorded for a verified run"}
              </dd>
            </div>
            <div>
              <dt>Runtime</dt>
              <dd>
                {research.run?.runtime ?? "Not recorded for a verified run"}
              </dd>
            </div>
            <div>
              <dt>Model revision</dt>
              <dd className="mono break-anywhere">
                {research.run?.revision ?? p.revision}
              </dd>
            </div>
          </dl>
          <Source path="configs/pope_llava.json" />
          <p className="caption">
            Raw model outputs and hidden states remain in the local run folder.
            Public pages are generated from verified aggregates. The site
            performs no model inference.
          </p>
          <a href={sourceUrl("docs/PROTOCOL.md")}>
            Read the full protocol and decision rules{" "}
            <span aria-hidden="true">↗</span>
          </a>
        </section>
      </article>
    </div>
  );
}
