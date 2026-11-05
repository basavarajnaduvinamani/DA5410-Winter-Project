import type { Metadata } from "next";
import { research } from "@/lib/data";
import { PageIntro, Source } from "@/components/shell";
import { Explorer } from "@/components/explorer";
export const metadata: Metadata = {
  title: "Example explorer",
  description:
    "Inspect saved POPE answers, image evidence, probe scores and gate traces. No live inference or invented examples.",
};
export default function ExplorerPage() {
  return (
    <div className="wide-page">
      <PageIntro
        eyebrow="INSPECT THE EVIDENCE"
        title="An answer is not the whole story."
      >
        <p>
          Look at the image, the exact question and both model answers. Filter
          for improvements and regressions; the same evidence standard applies
          to each.
        </p>
      </PageIntro>
      <Explorer
        examples={research.examples}
        hasRun={research.status === "measured"}
      />
      <div className="explorer-footnote">
        <p>
          Only COCO val2014 images with verified redistribution terms and
          complete attribution are eligible. Unknown gate traces remain unknown.
          No question is sent to a model from this page.
        </p>
        <p>
          The gallery is illustrative. Licence exclusions can hide fixes or
          regressions; a zero filter count does not mean zero such outcomes in
          the full test partition. Use the complete results tables for effect
          sizes.
        </p>
        <Source
          path={
            research.examples.length
              ? "results/examples_manifest.json"
              : "site/scripts/export-data.mjs"
          }
          label="Example provenance: the build-time export contract"
        />
      </div>
    </div>
  );
}
