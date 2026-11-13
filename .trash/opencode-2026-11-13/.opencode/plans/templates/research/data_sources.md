### Purpose
Document the origin, quality, and access constraints of all data used in the research to preserve provenance and reproducibility.

### Required-content checklist
- [ ] Each dataset/source is named with provenance details.
- [ ] Access method, permissions, and retention limits are recorded.
- [ ] Data quality caveats and known biases are listed.
- [ ] Versioning strategy (snapshot/hash/date) is specified.

### Drafter prompts
- What is the authoritative source of truth for each input?
- Which data elements could introduce confounding factors?
- How will another contributor rehydrate the exact same data slice?

### Example
Use `customer-events-v5` from the governed warehouse (`snapshot=2026-05-15`) plus synthetic perturbation set `spike-noise-r2`. Record warehouse query hash, row-count checksum, and PII redaction policy reference.
