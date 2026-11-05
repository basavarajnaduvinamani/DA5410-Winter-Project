# Preserve evidence hashes across Windows and Linux

## Context
Pre-push byte comparisons found that Git core.autocrlf normalized a generated metrics JSON file while its published manifest recorded the Windows CRLF hash. A clean Linux build would correctly reject that mismatch. PDF cross-reference offsets must also survive checkout unchanged.

## Options
A. Generate metrics as UTF-8/LF, preserve binary PDFs, and distinguish byte-exact artifact hashes from normalized source-text hashes.
B. Treat every hashed source file as binary, retaining operating-system-specific line endings.

## Decision
Use A. Metrics writers explicitly emit LF. Git attributes retain LF for published JSON and binary bytes for PDF/PNG. The public export digest remains byte-exact. Source-text digests explicitly normalize CRLF to LF so a Windows checkout and GitHub's LF source identify the same text. Upstream POPE question files remain byte-exact under their existing -text rule.

## Consequences
No scientific parameter, split, score or confidence interval changes. Raw run-artifact hashes remain raw. A regression checks that LF and CRLF sources share their normalized digest while their raw digests differ. Verify committed metrics bytes against the published manifest before pushing.
