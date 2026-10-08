# Versioned publication releases

This public repository contains the curated replication package. The separate development repository and its historical commits are private and are not part of this repository or its release archives.

For each publication release:

1. Reproduce the quick report and Julia tests, review the changes, and verify the distribution with `python3 scripts/check_release.py`.
2. Update the version and actual release date in `CITATION.cff`, then deliberately regenerate `provenance/FILE_SHA256.json` after reviewing the final files.
3. Enable this public repository in the existing Zenodo GitHub integration before creating the first GitHub release.
4. Create a GitHub release at the reviewed commit. Verify that Zenodo archives it and assigns a version-specific DOI.
5. Cite that version-specific DOI in the manuscript. A concept DOI identifies the collection of versions. Add the resulting DOI link to the GitHub README and citation metadata in a subsequent documentation commit without moving the archived release tag.
6. Changes to numerical code or evidence require a new release version; do not replace the source corresponding to the manuscript's cited DOI.

Use `CITATION.cff` as the metadata source unless Zenodo-specific fields are necessary. If `.zenodo.json` is added, Zenodo gives it precedence, so keep the two consistent.

Sources: [Zenodo GitHub releases](https://help.zenodo.org/docs/github/archive-software/github-upload/), [Zenodo citation metadata](https://help.zenodo.org/docs/github/describe-software/citation-file/).
