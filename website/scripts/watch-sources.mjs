import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// Content is generated, so watch its authoritative inputs outside Astro's root.
export default function watchSources() {
  const repo = fileURLToPath(new URL('../../', import.meta.url));
  const website = path.join(repo, 'website');
  return {
    name: 'particula-source-watch',
    configureServer(server) {
      const roots = ['docs', 'particula', 'website/results', 'website/templates', 'website/examples.json', 'website/AUTHORING.md', 'scripts/docs_site.py', 'scripts/docs_examples.py']
        .map((p) => path.join(repo, p));
      server.watcher.add(roots);
      let timer;
      function changed(file) {
        if (!roots.some((root) => file === root || file.startsWith(root + path.sep))) return;
        clearTimeout(timer);
        timer = setTimeout(() => {
          const result = spawnSync(process.execPath, ['scripts/python.mjs', '../scripts/docs_site.py', 'prepare'], { cwd: website, stdio: 'inherit' });
          if (result.status === 0) server.ws.send({ type: 'full-reload' });
          else server.config.logger.error('Source preparation failed; fix the reported error before previewing.');
        }, 300);
      }
      server.watcher.on('change', changed).on('add', changed).on('unlink', changed);
      server.httpServer?.once('close', () => clearTimeout(timer));
    },
  };
}
