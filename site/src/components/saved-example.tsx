"use client";

import { useState } from "react";
import Image from "next/image";
import type { Example } from "@/lib/types";

export function SavedExample({ example }: { example: Example }) {
  const [steered, setSteered] = useState(false);

  return (
    <section id="example" className="saved-example" aria-labelledby="example-title">
      <h2 id="example-title">One saved answer change</h2>
      <div className="saved-case">
        <figure>
          <Image src={example.imageSrc} alt={example.imageAlt} width={360} height={260} preload />
          <figcaption><a href={example.attribution.source}>{example.attribution.creator}</a>{" / "}<a href={example.attribution.licenseUrl}>Photo licence</a></figcaption>
        </figure>
        <div>
          <p className="case-question">{example.question}</p>
          <p className="case-label">Dataset answer: {example.truth === "no" ? "No" : "Yes"}</p>
          <div className="case-toggle" role="group" aria-label="Saved LLaVA run">
            <button type="button" aria-pressed={!steered} onClick={() => setSteered(false)}>Original</button>
            <button type="button" aria-pressed={steered} onClick={() => setSteered(true)}>With steering</button>
          </div>
          <p className="case-answer" aria-live="polite">Model answer <strong>{steered ? example.steered : example.baseline}</strong></p>
        </div>
      </div>
      <dl className="case-scores" aria-live="polite">
        <div><dt>Model confidence</dt><dd>Unavailable</dd></div>
        <div><dt>Probe score</dt><dd>{steered ? "Unavailable" : example.probeScore?.toPrecision(6) ?? "Unavailable"}</dd></div>
      </dl>
      <p className="case-label">The image stays fixed. Steering changes an internal model state. The public stream-swap record contains aggregate measurements.</p>
    </section>
  );
}
