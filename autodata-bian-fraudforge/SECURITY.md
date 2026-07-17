# Security policy

## Reporting

Please open a private security advisory for vulnerabilities involving code execution, dependency
compromise, secret leakage, unsafe file handling, or denial of service. Never include real bank
data, account identifiers, authentication material, or payment credentials in an issue.

## Demo boundaries

- The shipped generator creates fictional records and identifiers.
- The default app requires no network access or secrets.
- Uploaded transaction support is intentionally absent from the Gradio demo; the API accepts only
  explicit JSON features.
- Production deployment should add authentication, rate limits, audit logging, retention controls,
  and institution-specific privacy review.
