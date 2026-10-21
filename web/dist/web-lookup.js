// Fetch plain-text reference material directly; no API key or inference server.
const API = 'https://en.wikipedia.org/w/api.php';
const filler = new Set('a an the i me my you your we our it its is are was were be been do does did can could would should will please tell explain describe define what who where when why how about of to for in on and or this that mean meaning know want understand'.split(' '));

export function lookupQuery(input, unknown = []) {
  const words = input.match(/[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)*/gu) || [];
  const keywords = words.filter(word => !filler.has(word.toLowerCase()));
  const terms = keywords.length ? keywords : unknown;
  return terms.join(' ').slice(0, 240);
}

export async function lookupWeb(input, unknown, { signal, fetchImpl = fetch } = {}) {
  const query = lookupQuery(input, unknown);
  if (!query) throw new Error('Try naming the topic you want to look up.');
  const timeout = AbortSignal.timeout(12000);
  const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
  const search = async (terms) => {
    const params = new URLSearchParams({
      action: 'query', format: 'json', formatversion: '2', origin: '*',
      generator: 'search', gsrsearch: terms, gsrnamespace: '0', gsrlimit: '3',
      prop: 'extracts', exintro: '1', explaintext: '1', exchars: '650', exlimit: '3',
    });
    const response = await fetchImpl(`${API}?${params}`, {
      signal: requestSignal, credentials: 'omit', referrerPolicy: 'no-referrer',
    });
    if (!response.ok) throw new Error('Wikipedia is unavailable. Please try again later.');
    const data = await response.json();
    if (data.error) throw new Error('Wikipedia could not complete this search. Try a shorter topic.');
    return (data.query?.pages || []).filter(page =>
      typeof page.title === 'string' && typeof page.extract === 'string' && page.extract.trim()
    ).sort((a, b) => (a.index ?? 99) - (b.index ?? 99));
  };
  let pages = await search(query);
  // A long question may not match; retry only its unfamiliar terms once.
  const fallback = lookupQuery(unknown.join(' '));
  if (!pages.length && fallback && fallback !== query) pages = await search(fallback);
  if (!pages.length) return { query, text: '', sources: [] };
  return {
    query,
    text: pages[0].extract.trim().slice(0, 700),
    sources: pages.map(page => ({
      title: page.title,
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replaceAll(' ', '_'))}`,
    })),
  };
}
