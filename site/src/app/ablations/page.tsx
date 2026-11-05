import type { Metadata } from "next";
import { research } from "@/lib/data";
import { PageIntro, Source, PendingNote } from "@/components/shell";
import { SteeringTable, ControlComparisonTable } from "@/components/results";
import { FalseYesChart } from "@/components/charts";
export const metadata: Metadata = {
  title: "Ablations and controls",
  description:
    "Every planned steering control, its selection rule and measured trade-off, including negative findings.",
};
export default function Ablations() {
  return (
    <div className="wide-page">
      <PageIntro
        eyebrow="CONTROLS / ALTERNATIVE EXPLANATIONS"
        title="A reduction is not enough."
      >
        <p>
          A useful direction should do more than a matched random vector or
          shuffled labels. Recognition loss must stay visible beside fewer
          false-yes answers.
        </p>
      </PageIntro>
      {research.status === "pending" ? (
        <PendingNote title="No measured ablation results">
          <p>
            The comparisons below are the registered protocol. No control,
            effect size or confidence interval has been filled without a saved
            model run.
          </p>
        </PendingNote>
      ) : (
        <>
          <SteeringTable />
          <ControlComparisonTable />
          <FalseYesChart headingLevel={2} />
        </>
      )}
      <div className="ablation-list">
        {[
          [
            "Unsteered baseline",
            "No hook; greedy decoding. This is the common reference for every paired comparison.",
          ],
          [
            "SLERP toward the PCA direction",
            "Norm-preserving rotation. The layer, direction and cosine gate use calibration; strength uses validation.",
          ],
          [
            "Mean-difference addition",
            "Add a calibration mean-difference direction, scaled by mean activation norm. Its strength has a separate validation sweep.",
          ],
          [
            "Random-direction control",
            "A seeded random direction with the selected SLERP strength. It tests whether perturbation alone explains the effect.",
          ],
          [
            "Shuffled-label control",
            "A mean-difference direction after permuting calibration labels, with the same SLERP strength. PCA is not used here because its direction largely ignores labels.",
          ],
        ].map(([title, body], i) => (
          <section key={title}>
            <span className="section-index">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div>
              <h2>{title}</h2>
              <p>{body}</p>
            </div>
          </section>
        ))}
      </div>
      <Source path="src/pope_pipeline.py" />
      <section className="paper-section compact">
        <h2>How to read a negative result</h2>
        <p>
          If an interval crosses the comparison boundary, the finding is
          inconclusive. If a control matches SLERP, the proposed geometric
          mechanism is not established. If hallucination falls while recall
          falls too, report both; do not rename the trade-off a safety
          improvement.
        </p>
        <Source path="docs/PROTOCOL.md" />
      </section>
      <section className="paper-section compact">
        <p className="eyebrow">ARCHITECTURE CHECK</p>
        <h2>A newer model is a separate replication.</h2>
        <p className="mono break-anywhere">{research.modern.model}</p>
        <p>
          Baseline and detection only, with the same question files, partitions
          and scorer. This adapter is stub-tested; actual model execution and T4
          memory fit remain unverified.
        </p>
        <p className="status-text">Status: {research.modern.status}</p>
        <Source path={research.modern.source} />
      </section>
    </div>
  );
}
