# GitHub release checklist

## Before making the repository public

- [ ] Replace `OWNER` in `pyproject.toml` with the GitHub account or organization.
- [ ] Confirm the project name does not imply endorsement by the MixSIAR maintainers.
- [ ] Confirm the copyright holder and maintainer email.
- [ ] Review the provenance and redistribution terms of every file in `data/` and `reference_r/`.
- [ ] Add the complete GPL-3.0 license text if the hosting workflow does not generate it automatically.
- [ ] Run `py -3.9 -m pytest -q` from a fresh virtual environment.
- [ ] Check that GitHub Actions passes on Windows and Linux.
- [ ] Confirm that `outputs/`, environments, caches and posterior NetCDF files are not committed.
- [ ] Execute and archive the cross-language validation matrix described in `VALIDATION.md`.
- [ ] Add author-approved citation metadata (`CITATION.cff`) and a changelog.
- [ ] Create a tagged release and archive it with Zenodo to obtain a software DOI.
- [ ] Replace manuscript placeholders with the repository URL, release tag and DOI.

## Suggested first release sequence

1. Publish the repository privately and let CI run.
2. Invite one statistical reviewer and one new Python user to test installation and the wolves workflow.
3. Resolve licensing and attribution questions.
4. Complete the R/JAGS versus PyMC validation benchmark.
5. Tag a validation candidate, archive it, and only then cite that immutable release in the manuscript.
