# Authenticate protected preview QA without persisting credentials

## Context
Vercel previews redirect unauthenticated clients to platform login. The research site itself has no application authentication. Remote Playwright checks must exercise the exact preview without disabling deployment protection or exposing credentials in reports.

## Options
A. Inject a short-lived development OIDC token through the CLI and attach it only to the preview origin.
B. Disable deployment protection or introduce a long-lived bypass secret.

## Decision
Use A. The CLI injects credentials into process memory. Browser interception attaches the Trusted Sources header only when request origin matches BASE_URL. API checks attach it only for that origin and do not follow redirects. No token is stored in test configuration or source. Authenticated remote traces are disabled because network traces could persist the header; every assertion remains enabled. Local/CI runs keep failure traces.

## Consequences
Same-project development-to-preview access uses the existing Vercel default. No Trusted Sources rules are broadened. A failed token produces a failing check, not a request to turn off protection. Raw QA outputs remain ignored, and public summaries contain counts and hashes only.
