import Image from "next/image";
import type { Example } from "@/lib/types";

export function SavedExample({ example }: { example: Example }) {
  const score = example.probeScore;

  return (
    <section id="example" className="saved-example redesign-section" aria-labelledby="example-title">
      <h2 id="example-title">Steering example</h2>
      <div className="saved-case">
        <figure>
          <Image src={example.imageSrc} alt={example.imageAlt} width={360} height={260} loading="lazy" />
          <figcaption><a href={example.attribution.source}>{example.attribution.creator}</a>{" / "}<a href={example.attribution.licenseUrl}>Photo licence</a></figcaption>
        </figure>
        <div>
          <p className="case-question">{example.question}</p>
          <p className="case-label">Dataset answer: {example.truth === "no" ? "No" : "Yes"}</p>
          <p className="case-answer">Original model answer <strong>{example.baseline}</strong></p>
          <p className="case-label">Original probe score: {score === null ? "Unavailable" : score > 0 && score < 0.001 ? "< 0.001" : score.toFixed(3)}</p>
          <details className="case-toggle">
            <summary>Show answer with steering</summary>
            <p className="case-answer">Model answer <strong>{example.steered}</strong></p>
          </details>
        </div>
      </div>
      <p className="case-label">The image stays fixed. Steering changes an internal model state.</p>
    </section>
  );
}
