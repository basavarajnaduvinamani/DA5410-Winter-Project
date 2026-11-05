import { research, metric, profileLabel, value } from "@/lib/data";
import { PendingNote, Source } from "./shell";
const detectors = [
  ["probe", "Linear probe"],
  ["mean_difference", "Mean-difference projection"],
  ["pca", "PCA projection"],
  ["neg_logprob", "Answer-token −log p"],
  ["entropy", "Answer-token entropy"],
];
export const methods = [
  ["slerp_pca", "SLERP · PCA direction"],
  ["add_mean_difference", "Mean-difference addition"],
  ["random_direction", "Random-direction control"],
  ["shuffled_labels", "Shuffled-label control"],
];
export function ProfileCaption() {
  return (
    <p className="profile-caption">
      {profileLabel()}
      <span> · image-clustered confidence intervals</span>
    </p>
  );
}
export function BaselineTable() {
  if (!research.run) return null;
  return (
    <>
      <ProfileCaption />
      <div
        className="table-scroll"
        role="region"
        aria-label="POPE baseline results, horizontally scrollable"
        tabIndex={0}
      >
        <table>
          <caption>Unsteered POPE baseline</caption>
          <thead>
            <tr>
              <th scope="col">POPE split</th>
              <th scope="col">Questions</th>
              <th scope="col">Accuracy [CI]</th>
              <th scope="col">Precision [CI]</th>
              <th scope="col">Recall [CI]</th>
              <th scope="col">F1 [CI]</th>
              <th scope="col">Yes-ratio [CI]</th>
            </tr>
          </thead>
          <tbody>
            {["random", "popular", "adversarial", "pooled"].map((split) => {
              const r = research.results.baseline[`test:${split}`];
              return r ? (
                <tr key={split}>
                  <th scope="row">{split}</th>
                  <td>{r.n}</td>
                  <td>{metric(r.accuracy)}</td>
                  <td>{metric(r.precision)}</td>
                  <td>{metric(r.recall)}</td>
                  <td>{metric(r.f1)}</td>
                  <td>{metric(r.yes_ratio)}</td>
                </tr>
              ) : null;
            })}
          </tbody>
        </table>
      </div>
      <Source path="results/REPORT_metrics.json" />
    </>
  );
}
export function DetectionTable() {
  const d = research.results.detection;
  if (!d) return null;
  return (
    <>
      <ProfileCaption />
      <div
        className="table-scroll"
        role="region"
        aria-label="Detection results, horizontally scrollable"
        tabIndex={0}
      >
        <table>
          <caption>Frozen detectors on unseen images</caption>
          <thead>
            <tr>
              <th scope="col">Detector</th>
              <th scope="col">Selected layer</th>
              <th scope="col">AUROC [CI]</th>
              <th scope="col">AUPRC [CI]</th>
            </tr>
          </thead>
          <tbody>
            {detectors.map(([key, label]) => (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>{research.selectedLayers[key] ?? "No fitted layer"}</td>
                <td>{metric(d.detectors[key]?.auroc)}</td>
                <td>{metric(d.detectors[key]?.auprc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Source path="results/REPORT_metrics.json" />
    </>
  );
}
export function SteeringTable() {
  if (!research.run) return null;
  return (
    <>
      <ProfileCaption />
      <div
        className="table-scroll"
        role="region"
        aria-label="Steering results, horizontally scrollable"
        tabIndex={0}
      >
        <table>
          <caption>Steering and matched controls</caption>
          <thead>
            <tr>
              <th scope="col">Intervention</th>
              <th scope="col">Strength</th>
              <th scope="col">False-yes [CI]</th>
              <th scope="col">Δ false-yes [CI]</th>
              <th scope="col">Recall [CI]</th>
              <th scope="col">Retention [CI]</th>
              <th scope="col">Δ recall [CI]</th>
            </tr>
          </thead>
          <tbody>
            {methods.map(([key, label]) => {
              const r = research.results.steering[key];
              return (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  <td>{r ? value(r.strength) : "Not run"}</td>
                  <td>{metric(r?.false_yes_rate)}</td>
                  <td>{metric(r?.delta_false_yes_rate)}</td>
                  <td>{metric(r?.recall)}</td>
                  <td>{metric(r?.retention)}</td>
                  <td>{metric(r?.delta_recall)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Δ is method minus baseline; a negative false-yes delta favours steering.
        Retention is the fraction of baseline-correct answers that remain
        correct, distinct from recall.
      </p>
      <Source path="results/REPORT_metrics.json" />
    </>
  );
}
export function ResultsOverview() {
  if (research.status === "pending")
    return (
      <PendingNote title="No verified model results yet">
        <p>{research.statusDetail}</p>
        <p>
          AUROC, false-yes reduction and recall retention are not available. No
          performance values or example answers are filled by hand.
        </p>
        <Source path="results/2026-11-01/gpu_status.json" />
      </PendingNote>
    );
  return (
    <>
      <div className="endpoint-note">
        <strong>
          Endpoint:{" "}
          {research.run?.target === "error"
            ? "answer error"
            : "false-yes hallucination"}
          .
        </strong>{" "}
        {research.run?.target === "error"
          ? "This detector includes wrong “no” answers. Its AUROC must not be described as false-yes-only detection."
          : "The detector target is a yes answer about an absent object."}
      </div>
      <BaselineTable />
      <DetectionTable />
      <SteeringTable />
    </>
  );
}
export function ControlComparisonTable() {
  const rows = research.results.steering.slerp_pca?.versus_controls_false_yes;
  if (!rows) return null;
  return (
    <>
      <ProfileCaption />
      <div
        className="table-scroll"
        role="region"
        aria-label="Paired control comparisons, horizontally scrollable"
        tabIndex={0}
      >
        <table>
          <caption>Does SLERP outperform the controls?</caption>
          <thead>
            <tr>
              <th scope="col">Comparison</th>
              <th scope="col">Δ false-yes [CI]</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(rows).map(([key, m]) => (
              <tr key={key}>
                <th scope="row">
                  SLERP minus{" "}
                  {key === "random_direction"
                    ? "random direction"
                    : "shuffled labels"}
                </th>
                <td>{metric(m)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Negative favours SLERP. An interval including zero leaves superiority
        inconclusive. This is a paired comparison, distinct from comparing
        overlapping marginal intervals.
      </p>
      <Source path="results/REPORT_metrics.json" />
    </>
  );
}
