// The API plugin emits complete symbol sections as HTML inside MarkdownContent.
// Preserve its markup and place each curated reference after the exact heading.
export function apiStudySections(content, links) {
  const bySymbol = Map.groupBy(links, (link) => link.symbol);
  const sections = [];
  let start = 0;
  for (const match of content.matchAll(/<h([1-6])\b[^>]*\bid="([^"]+)"[^>]*>[\s\S]*?<\/h\1>/gi)) {
    const related = bySymbol.get(match[2]);
    if (!related) continue;
    const end = match.index + match[0].length;
    sections.push({ html: content.slice(start, end), related });
    start = end;
    bySymbol.delete(match[2]);
  }
  if (bySymbol.size) throw new Error('Missing generated API symbol headings: ' + [...bySymbol.keys()].join(', '));
  sections.push({ html: content.slice(start), related: [] });
  return sections;
}
