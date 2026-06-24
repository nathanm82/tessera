# Security Policy

## Supported versions

This project is pre-1.0; fixes land on the latest released minor version.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | yes       |
| < 0.3   | no        |

## Reporting a vulnerability

Please report security issues privately using GitHub's **Report a vulnerability**
button under the Security tab, rather than opening a public issue.

Include a description, a minimal reproduction, and the affected version. You can
expect an initial response within a few days. Once a fix is available it will be
released and the report disclosed.

## Scope notes

`tessera` reads image files from paths you supply and can load optional third-party
model weights through the `clip` extra. Treat corpora and model checkpoints from
untrusted sources with the same caution you would any external input.
