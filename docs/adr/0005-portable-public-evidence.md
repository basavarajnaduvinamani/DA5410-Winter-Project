# Package the public evidence boundary for Vercel

## Context
The requested Vercel CLI deployment runs from site/. The exporter normally reads protocol and verified aggregates from the parent research repository, which is intentionally outside the upload boundary. Uploading the entire checkout risks including private working artifacts.

## Options
A. Regenerate locally and package the public JSON with its SHA-256 and source hashes; validate that package during standalone builds.
B. Upload the whole research repository and configure a monorepo root directory.
C. Deploy only a prebuilt out/ directory and omit remote build checks.

## Decision
Use A. When research sources exist, the committed exporter generates both identical JSON copies and a public evidence manifest. When sources are outside the deployment package, it checks both JSON copies against the manifest and validates their schema. Vercel receives site/ only. The manifest identifies source bytes, not independent model-weight or image-byte verification.

## Consequences
Standalone builds are reproducible without research caches, private files or model weights. A research change requires regeneration before deployment. Hashes provide consistency and provenance within a reviewed Git commit, not cryptographic attestation of the experiment itself.
