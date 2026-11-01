import type { Metadata } from "next";
import { repo } from "@/lib/data";
import { PageIntro, Source } from "@/components/shell";
export const metadata: Metadata = {
  title: "About and licensing",
  description:
    "Research context, credits, evidence boundaries and privacy for CounterPrior.",
};
export default function About() {
  return (
    <div className="narrow-page">
      <PageIntro eyebrow="PEOPLE / RESEARCH CONTEXT" title="About CounterPrior">
        <p>
          CounterPrior is research on object hallucination and
          representation-level intervention in vision-language models, conducted
          at AI4Bharat Lab, IIT Madras.
        </p>
      </PageIntro>
      <section className="about-section">
        <h2>Basavaraj A Naduvinamani</h2>
        <p>
          Joint Master&apos;s student in Data Science & AI at IIT Madras and the
          University of Birmingham. Research interests include visual grounding,
          interpretable model behaviour and reliable evaluation.
        </p>
        <a href="https://github.com/basavarajnaduvinamani">
          Contact through GitHub <span aria-hidden="true">↗</span>
        </a>
        <dl className="credits-list">
          <div>
            <dt>Supervisor</dt>
            <dd>Prof. Mitesh M. Khapra</dd>
          </div>
          <div>
            <dt>Mentor</dt>
            <dd>Mohammed Safi Ur Rahman Khan</dd>
          </div>
          <div>
            <dt>Research setting</dt>
            <dd>AI4Bharat Lab, IIT Madras</dd>
          </div>
        </dl>
        <p className="caption">
          Affiliation and credit are provided by the author. Institution names
          do not imply endorsement of this site.
        </p>
      </section>
      <section id="licensing" className="about-section">
        <h2>Licensing and privacy</h2>
        <p>
          POPE question files retain the upstream MIT licence. COCO photographs
          have individual Flickr terms; this site publishes an example only with
          recorded redistribution rights and attribution. A model licence does
          not grant rights to its training images.
        </p>
        <p>
          No blanket open-source licence is granted to the repository by this
          website. Consult the source and contact the author before reuse.
          Registered Qwen replication and LLaVA remain subject to their model
          licences.
        </p>
        <Source path="dataset/pope/LICENSE_POPE" />
        <p>
          This is a static site: no accounts, analytics, cookies, uploads,
          database or model inference. A theme preference may be stored locally
          in your browser. Vercel serves the pages and may process standard
          hosting request logs; this site adds no visitor tracking.
        </p>
      </section>
      <section className="about-section">
        <h2>A record that can be corrected.</h2>
        <p>
          The earlier custom benchmark and headline claims are preserved in the
          repository with their corrections. Proposed relabels remain proposals.
          New findings will link to the exact saved run and evaluation script,
          including failures and regressions.
        </p>
        <a href={`${repo}/issues`}>
          Report a mistake <span aria-hidden="true">↗</span>
        </a>
      </section>
    </div>
  );
}
