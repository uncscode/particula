import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const local = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));
const python = process.env.PARTICULA_DOCS_PYTHON || (existsSync(local) ? local : 'python3');
const result = spawnSync(python, process.argv.slice(2), { stdio: 'inherit' });
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
