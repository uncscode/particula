// Emit escaped diagram source; the page loads Mermaid only when needed.
export default function remarkMermaid() {
  return function transform(tree) {
    function visit(node) {
      if (node.type === 'code' && node.lang === 'mermaid') {
        node.type = 'html';
        node.value = '<pre class="mermaid">' + node.value
          .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') + '</pre>';
        delete node.lang;
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}
