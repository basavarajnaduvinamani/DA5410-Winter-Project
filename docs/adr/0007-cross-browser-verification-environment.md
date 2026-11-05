# Use Ubuntu for the complete cross-browser release gate

## Context
Windows Chromium functional checks and axe scans passed, while Firefox intermittently stalled during context disposal after passing page assertions. Reducing concurrency, disabling tracing for diagnosis and explicitly releasing the page did not resolve it. The temporary cleanup workaround was removed. An existing Ubuntu 24.04 WSL environment ran every unchanged Firefox check successfully.

## Options
A. Keep retrying Windows until a lucky run passes.
B. Run the unchanged complete suite in Ubuntu, matching the GitHub Actions platform.
C. Skip Firefox or relax its assertions.

## Decision
Use B. Keep assertions, console-error checks, accessibility checks and failure tracing. Install the official Playwright browser dependencies in the existing Ubuntu environment and use a checksum-verified Node runtime. Keep local concurrency at one worker. CI uses Ubuntu and the same lockfile and Playwright configuration.

## Consequences
Linux results establish the tested browser-engine behavior; they do not erase the recorded Windows runner failures. Reports identify the test OS. Browser binaries and installation logs are excluded from Git. Production still requires a complete passing suite, the screenshot review and mobile Lighthouse scores.
