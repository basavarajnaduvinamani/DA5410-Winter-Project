"use client";
import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import type { Example } from "@/lib/types";
import { filterExamples } from "@/lib/explorer";
export function Explorer({
  examples,
  hasRun = false,
}: {
  examples: Example[];
  hasRun?: boolean;
}) {
  const [split, setSplit] = useState("all"),
    [filter, setFilter] = useState("all"),
    [selected, setSelected] = useState<string | null>(null);
  const visible = filterExamples(examples, split, filter);
  const example = visible.find((e) => e.id === selected) ?? visible[0];
  return (
    <div className="explorer">
      <div className="explorer-controls">
        <div className="field">
          <label htmlFor="pope-split">POPE split</label>
          <select
            id="pope-split"
            value={split}
            onChange={(e) => setSplit(e.target.value)}
          >
            <option value="all">All splits</option>
            <option value="random">Random</option>
            <option value="popular">Popular</option>
            <option value="adversarial">Adversarial</option>
          </select>
        </div>
        <div className="filter-field">
          <span id="outcome-label">Answer outcome</span>
          <div
            className="filter-group"
            role="group"
            aria-labelledby="outcome-label"
          >
            {[
              ["all", "All outcomes"],
              ["hallucinated", "Baseline hallucinated"],
              ["fixed", "Fixed by steering"],
              ["broken", "Broken by steering"],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="result-count" role="status" aria-live="polite">
        {visible.length} {visible.length === 1 ? "example" : "examples"} match
        your filters
      </p>
      {!examples.length ? (
        <div className="empty-explorer">
          <span className="empty-symbol" aria-hidden="true">
            <svg viewBox="0 0 80 64" fill="none" stroke="currentColor">
              <rect x="8" y="8" width="64" height="44" rx="5" />
              <path d="M24 58h32M40 52v6M23 25h34M23 34h23" />
            </svg>
          </span>
          <h2>
            {hasRun
              ? "Image attribution is not verified yet"
              : "Results land after the GPU run"}
          </h2>
          <p>
            No example answers or images have been published. The explorer will
            populate from a verified run and licence-checked COCO metadata, not
            demonstration content.
          </p>
          <Link href="/#reproduce">
            See the reproducibility steps <span aria-hidden="true">→</span>
          </Link>
        </div>
      ) : !example ? (
        <div className="empty-explorer">
          <h2>No matching examples</h2>
          <p>Try another split or answer outcome.</p>
          <button
            type="button"
            onClick={() => {
              setSplit("all");
              setFilter("all");
            }}
          >
            Reset filters
          </button>
        </div>
      ) : (
        <>
          <div className="field example-picker">
            <label htmlFor="example">Choose an example</label>
            <select
              id="example"
              value={example.id}
              onChange={(e) => setSelected(e.target.value)}
            >
              {visible.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.split} · COCO {e.imageId} · {e.question}
                </option>
              ))}
            </select>
          </div>
          <div className="example-grid">
            <figure>
              <Image
                src={example.imageSrc}
                alt={example.imageAlt}
                width={640}
                height={480}
                className="example-image"
              />
              <figcaption>
                {example.attribution.title ? (
                  <>
                    {example.attribution.title}
                    <br />
                  </>
                ) : null}
                COCO val2014 · {example.attribution.creator}
                <br />
                <a href={example.attribution.source}>Flickr source</a> ·{" "}
                <a href={example.attribution.licenseUrl}>
                  {example.attribution.license}
                </a>
                <br />
                Original COCO image bytes; no image edits.
              </figcaption>
            </figure>
            <div className="example-answer">
              <p className="eyebrow">
                {example.split} / ground truth: {example.truth}
              </p>
              <h2>{example.question}</h2>
              <dl className="answer-list">
                <div>
                  <dt>Unsteered baseline</dt>
                  <dd>{example.baseline}</dd>
                </div>
                <div>
                  <dt>SLERP answer</dt>
                  <dd>{example.steered}</dd>
                </div>
              </dl>
              <div className="score">
                <h3>Probe: answer-error risk</h3>
                {example.probeScore === null ? (
                  <p>Score not recorded.</p>
                ) : (
                  <>
                    <div
                      className="score-track"
                      role="img"
                      aria-label={`Probe score ${example.probeScore.toFixed(3)}, threshold ${example.probeThreshold}`}
                    >
                      <span style={{ width: `${example.probeScore * 100}%` }} />
                      <i style={{ left: `${example.probeThreshold * 100}%` }} />
                    </div>
                    <p className="mono">
                      {example.probeScore.toFixed(3)}{" "}
                      <span>· threshold {example.probeThreshold}</span>
                    </p>
                  </>
                )}
                <p className="caption">
                  The probe cutoff is separate from the cosine steering gate.
                </p>
              </div>
              <div className="gate-detail">
                <strong>
                  Gate decision:{" "}
                  {example.gate.triggered === null
                    ? "not recorded"
                    : example.gate.triggered
                      ? "steered"
                      : "not steered"}
                </strong>
                <p>
                  {example.gate.triggered === null
                    ? "This run contains no gate trace. A probe score is not used to invent one."
                    : `Prefill cosine alignment ${example.gate.alignment?.toFixed(3)}, gate threshold ${example.gate.threshold?.toFixed(3)}.`}
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
