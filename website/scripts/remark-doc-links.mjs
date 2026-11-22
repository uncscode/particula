import path from 'node:path';
import topics from '../topics.json' with { type: 'json' };

const studyFigures = new Map(topics.resources
  .filter((resource) => resource.tocFigure)
  .map(({ source, tocFigure }) => [
    source.replace(/^docs\//, '').replace(/\.py$/, '.md'),
    '/particula/images/' + tocFigure,
  ]));
const tocImages = new Set(studyFigures.values());

// Python examples can have an execution banner before their authored opening.
function groupStudyIntro(tree, figureUrl) {
  if (!figureUrl) return;
  const index = tree.children.findIndex(node => node.type === 'paragraph'
    && node.children[0]?.type === 'image' && node.children[0].url === figureUrl);
  if (index < 0) return;
  const figure = tree.children[index];
  const abstract = tree.children[index + 1];
  if (abstract?.type !== 'paragraph' || abstract.children[0]?.type === 'image') return;
  tree.children.splice(index, 2, {
    type: 'studyIntro',
    data: { hName: 'section', hProperties: { className: ['study-intro'], ariaLabel: 'Study summary' } },
    // Place the float first so the paragraph can flow beneath it.
    children: [figure, abstract],
  });
}

// Resolve Markdown links (including multiline labels and reference definitions)
// against their generated source location. Code nodes are deliberately untouched.
export default function remarkDocLinks() {
  return (tree, file) => {
    const filename = String(file.path ?? '').replaceAll('\\', '/');
    const marker = '/src/content/docs/';
    if (!filename.includes(marker)) return;
    const source = filename.split(marker)[1];
    function visit(node) {
      if (['link', 'image', 'definition'].includes(node.type) && node.url
        && /^https?:\/\/uncscode\.github\.io\/particula(?:\/|[?#]|$)/i.test(node.url)) {
        const target = new URL(node.url);
        node.url = target.pathname + target.search + target.hash;
      }
      if (['link', 'image', 'definition'].includes(node.type) && node.url
        && !/^(?:[a-z]+:|\/|#)/i.test(node.url)) {
        const match = node.url.match(/^([^?#]+)(.*)$/);
        if (match && /\.(?:md|mdx|ipynb)$/.test(match[1])) {
          let target = path.posix.normalize(path.posix.join(path.posix.dirname(source), decodeURI(match[1])));
          target = target.replace(/(?:^|\/)index\.(?:md|mdx)$/, '').replace(/\.(?:md|mdx|ipynb)$/, '');
          node.url = '/particula/' + target.replace(/^\/+|\/+$/g, '') + '/' + match[2];
        }
      }
      // Display a 300 dpi PNG at the same three-inch width as a vector TOC.
      if (node.type === 'image' && tocImages.has(node.url)) {
        node.data ??= {};
        node.data.hProperties ??= {};
        Object.assign(node.data.hProperties, { width: 288, height: 288 });
      }
      node.children?.forEach(visit);
    }
    visit(tree);
    groupStudyIntro(tree, studyFigures.get(source));
    // MkDocs stripped emoji spacing and collapsed punctuation separators in
    // these published headings; retain their existing relocation destinations.
    const legacy = new Set([
      'initial-and-final-sizes-what-to-look-for',
      'time-evolution-what-to-look-for',
      'historical-terminology-unsupported-before-state',
    ]);
    const text = (node) => node.value ?? node.children?.map(text).join('') ?? '';
    function addAliases(parent) {
      if (!parent.children) return;
      parent.children = parent.children.flatMap((node) => {
      addAliases(node);
      if (node.type !== 'heading') return [node];
      const anchor = text(node).toLowerCase().replace(/[^\p{L}\p{N}_\s-]/gu, '')
        .trim().replace(/\s+/g, '-').replace(/-+/g, '-');
      return legacy.has(anchor)
        ? [{ type: 'html', value: `<a id="${anchor}"></a>` }, node]
        : [node];
      });
    }
    addAliases(tree);
  };
}
