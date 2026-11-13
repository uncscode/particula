// Translate block extensions using Markdown nodes and their source positions.
// Code fences and inline code are never searched or rewritten.
export default function remarkMkdocs() {
  const parser = this;
  return (tree, file) => {
    const source = String(file.value ?? '');
    function visit(parent) {
      if (!parent.children) return;
      const result = [];
      for (let i = 0; i < parent.children.length; i++) {
        const node = parent.children[i];
        if (node.type === 'heading') {
          const last = node.children.at(-1);
          const match = last?.type === 'text' && last.value.match(/\s*\{#([\w.-]+)\}\s*$/);
          if (match) {
            last.value = last.value.slice(0, match.index);
            node.data = { ...node.data, hProperties: { ...node.data?.hProperties, id: match[1] } };
          }
        }
        if (node.type === 'paragraph' && node.position) {
          const raw = source.slice(node.position.start.offset, node.position.end.offset);
          const marker = raw.match(/^(!!!|\?\?\?\+?|===)\s*(\w+)?\s*(?:"([^"]*)")?\s*\n?/);
          if (marker) {
            let body = raw.slice(marker[0].length).replace(/^ {4}/gm, '');
            const next = parent.children[i + 1];
            if (next?.type === 'code' && !next.lang && next.position
                && /^ {4}/m.test(source.slice(next.position.start.offset, next.position.end.offset))) {
              body += '\n\n' + next.value;
              i++;
            }
            const title = marker[3] || marker[2] || 'Note';
            const collapsible = marker[1].startsWith('???') || marker[1] === '===';
            const children = parser.parse(body).children;
            const heading = { type: 'paragraph', children: [{ type: 'text', value: title }],
              data: { hName: collapsible ? 'summary' : 'p', hProperties: { className: ['starlight-aside__title'] } } };
            result.push({ type: 'blockquote', children: [heading, ...children], data: {
              hName: collapsible ? 'details' : 'aside',
              hProperties: { className: ['starlight-aside', 'starlight-aside--note'],
                ...(marker[1] === '???+' ? { open: true } : {}) },
            } });
            continue;
          }
        }
        visit(node);
        result.push(node);
      }
      parent.children = result;
    }
    visit(tree);
  };
}
