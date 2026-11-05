# CounterPrior research site

Static Next.js App Router export, with no inference, service, auth, database or analytics.

## Local check
`npm ci`, `npm run prepare:examples`, `npm run export:data`, `npm run typecheck`, `npm run lint`, `npm run test`, `npm run build`, then `npm run test:e2e`. The last command starts the static host automatically. Install the Playwright Chromium, Firefox and WebKit binaries first. The local Node runtime is pinned; CI uses Node 24 LTS. The static asset step downloads only the licensed photographs listed in the committed public export and verifies every hash; photos stay outside Git.

## Publish evidence
Keep downloaded runs under ignored `incoming/`. Install `requirements-report.txt` in the project Python environment, then run `python -m scripts.ingest_run --zip PATH` from the repository root. It validates the run identity, registered config, pinned POPE bytes, disjoint image IDs, official scoring, coverage and validation-selected strengths before regenerating aggregates. It does not load a model. Its verification scope explicitly excludes rehashing original model weights and unseen COCO image bytes.

For images use `python -m scripts.export_site_examples --run RUN --annotations COCO_JSON --images COCO_FOLDER`. It refuses unknown redistribution terms or incomplete photographer attribution, copies at most the policy limit, and never invents gate traces. The original COCO annotation licence field alone is not a complete photographer credit. A run without eligible images keeps a truthful empty explorer.

The site exporter accepts measured metrics only when their hash matches `results/published_run.json`. No public demonstration answers or placeholder charts are generated. Synthetic fixtures live in tests only.

For a protected preview, set `BASE_URL` to its exact origin and run Playwright under `vercel env run` (`vercel.cmd` in PowerShell). The test helper attaches the short-lived development OIDC header only to that origin; authenticated traces are disabled so headers are not persisted. Local and CI failure traces remain enabled. Keep deployment protection on. Browser checks also compare the served evidence JSON byte-for-byte with the reviewed local export.

## Publication gate
The owner resumed the workflow after the preview checkpoint request. Complete local QA, deploy and test a preview, then publish production under the original approval. See the local task reports for actual verification and deployment status. A missing GPU run leaves the site in its explicit pending state.
