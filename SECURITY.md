# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Artaeon/exact-federated-vsa/security/advisories/new)
and include the affected revision, reproduction steps, expected impact, and any
suggested mitigation. Do not attach credentials, private datasets, or patient
records.

## Supported version

Security fixes target the latest revision on `main`. Older commits and generated
research artifacts are supported on a best-effort basis.

## Scope

Relevant reports include dependency or workflow compromise, unsafe deserialization,
path traversal, untrusted-download handling, and accidental inclusion of sensitive
data. Statistical limitations, model accuracy, and clinical suitability are research
questions rather than software vulnerabilities, but documentation corrections are
welcome through normal GitHub Issues.

This repository processes public TCGA-derived files for research. It is not designed
to ingest protected health information or controlled-access clinical data.
