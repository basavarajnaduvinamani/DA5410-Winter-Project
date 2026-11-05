# Compatible strict TypeScript and static QA toolchain

## Context
Official npm stable tags were checked at implementation time. The latest Next.js and React resolve to the exact versions in site/package-lock.json. Their stable ESLint dependencies report unsupported peer ranges for the newest ESLint and TypeScript major releases.

## Options
A. Ignore the peer constraints and use the newest majors anyway.
B. Keep the latest stable Next.js/React and select the newest compatible stable compiler/linter releases.
C. Replace the requested Next.js stack with a different static framework.

## Decision
Use B. Keep strict TypeScript enabled. Use direct current Lighthouse instead of the older LHCI dependency tree. Record exact installed versions and production dependency audit in site/qa. Never pass --force or --legacy-peer-deps to hide an incompatible dependency graph.

## Consequences
The compiler/linter may lag their newest stable majors until Next's parser dependencies support them. The site remains a static App Router export, with no server or secrets shipped. The lockfile, typecheck, lint, tests and build define the verified combination.
