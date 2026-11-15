import graph from '../../topics.json' with { type: 'json' };

export { graph };
export const topics = new Map(graph.topics.map((topic) => [topic.id, topic]));
export function topicPath(id, base = '/particula/') {
  return `${base.replace(/\/$/, '')}/guide/topics/${id === 'aerosols' ? '' : id + '/'}`;
}
export function ancestors(id) {
  const path = [];
  const seen = new Set();
  while (topics.has(id) && !seen.has(id)) {
    seen.add(id);
    const topic = topics.get(id);
    path.unshift(topic);
    id = topic.parent;
  }
  return path;
}
export function children(id) {
  return graph.topics.filter((topic) => topic.parent === id).sort((a, b) => a.order - b.order);
}
export function resourcePath(source, base = '/particula/') {
  const [file, fragment] = source.split('#');
  let route = file.replace(/^docs\//, '').replace(/(?:^|\/)index\.md$/, '').replace(/\.(?:md|py)$/, '');
  return `${base.replace(/\/$/, '')}/${route}${route ? '/' : ''}${fragment ? '#' + fragment : ''}`;
}
export function normalize(value) {
  return value.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}
export function searchTopics(query, limit = 6) {
  const term = normalize(query);
  if (!term) return [];
  const words = term.split(/\s+/);
  return graph.topics.map((topic) => {
    const label = normalize(topic.label);
    const aliases = topic.aliases.map(normalize);
    const phrases = topic.search_phrases.map(normalize);
    const text = normalize([topic.label, ...topic.aliases, ...topic.search_phrases, topic.description].join(' '));
    const score = label === term ? 100 : aliases.includes(term) ? 90 : phrases.includes(term) ? 80
      : label.includes(term) ? 70 : words.every((word) => text.includes(word)) ? 40 : 0;
    return { topic, score, path: ancestors(topic.id).map((t) => t.label).join(' → ') };
  }).filter(({ score }) => score > 0).sort((a, b) => b.score - a.score || a.topic.label.localeCompare(b.topic.label)).slice(0, limit);
}
