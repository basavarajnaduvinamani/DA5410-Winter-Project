# Keep photographs outside Git and verify publication assets

## Context
The owner forbids committing COCO images and requests at most 24 licensed, attributed COCO val2014 photographs served from the site's static folder. Standard COCO annotations record licences and Flickr image URLs, but do not provide a complete photographer credit.

## Options
A. Commit attribution and image hashes; fetch the exact bounded assets at build time.
B. Commit photographs alongside their attribution.
C. Publish no examples even when attribution can be verified.

## Decision
Use A. Verify the official annotation archive checksum, select deterministic illustrative cases (regressions, fixes, unchanged), and restrict licences to BY/BY-SA/BY-ND/public domain. BY-ND permits redistribution of unmodified material (https://creativecommons.org/licenses/by-nd/2.0/). Resolve photographer names and supplied titles through Flickr's public oEmbed metadata. Exclude failures. Save attribution evidence and hashes; fetch at most 24 original, unmodified image files into ignored static assets. Verify their hashes before publication. Gallery screenshots stay outside Git too. Vercel receives metadata and fetches the pinned images during its build.

## Consequences
No photographs enter Git. Clean builds need network access to the pinned public COCO assets and fail on a hash mismatch. The gallery is illustrative, never a statistical denominator. Unrecorded historical gate decisions remain unknown. Metadata does not independently establish that the exact same pixel bytes were used in the remote model run.
