# Static research publication with an explicit pending state

## Context
The owner requests a public research record, not live model inference. No verified pretrained-model output is available at implementation start. Latest owner steering requests a tested preview checkpoint and a halt before production.

## Options
A. Next.js App Router static export on Vercel, driven by saved run artifacts.
B. A server-rendered research dashboard with inference or a managed backend.
C. A buildless document site or Astro static site.

## Decision
Use A, the owner's specified stack. Read saved aggregates at build time, render the same typed export in all pages, and run no server, model, analytics, auth or database. Show a pending state without fake tables, charts, answers or images. Local UI theme preference is stored on the visitor's device only. Complete preview QA, then halt before production per the latest instruction.

## Consequences
A build is required to publish new results. Evidence is versioned and cacheable. A missing GPU run blocks scientific claims, not publication of the method and reproducibility protocol. The reviewer sees the endpoint and sample size beside every measured table. Deployment is gated by CPU, type, lint, build, unit, accessibility and cross-browser checks.
