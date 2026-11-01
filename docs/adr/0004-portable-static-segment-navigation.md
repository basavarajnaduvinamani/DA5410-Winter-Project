# Preserve App Router navigation on a plain static host

## Context
Browser tests of the Next.js static export observed segment prefetch URLs such as explorer/__next.explorer.__PAGE__.txt while the generated payload was explorer/__next.explorer/__PAGE__.txt. This produced console errors on a plain static host despite correct page HTML. The site must remain a static export.

## Options
A. Add deterministic build-time aliases for exported segment payloads, and serve them with the component content type.
B. Remove client navigation and use document-only anchors.
C. Introduce a server solely to handle segment routing.

## Decision
Use A. Copy only generated text payloads from __next.* directories into their requested flat alias names. Keep the original export. The local server and Vercel headers identify the payload content type. Add a browser test that follows the header links and asserts there are no console errors.

## Consequences
No server-side application, inference, auth or database is added. The alias step is version-dependent and must remain covered by navigation tests when Next.js is updated. Never silence console errors or remove accessibility checks to pass the QA gate.
