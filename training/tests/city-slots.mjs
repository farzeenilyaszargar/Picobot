import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createCitySlots } from '../../web/dist/city-slots.js';

const { aliases, cases } = JSON.parse(readFileSync(new URL('../picoboi_10m_city_cases.json', import.meta.url)));
for (const dialogue of cases) {
  const parser = createCitySlots(aliases);
  for (const [role, text, expected] of dialogue) {
    const segments = parser.encode(text, role);
    assert.equal(segments.map(part => part.text ?? `<city_${part.slot}>`).join(''), expected);
    assert.equal(segments.map(part => part.text ?? parser.resolve(part.slot).identity).join('').includes('undefined'), false);
    if (text.includes('Ignore <city_8>')) {
      assert.equal(parser.resolve(8), null);
      assert.match(segments.at(-1).text, /<city_8>/);
      assert.equal(parser.resolve(2).name, 'Auckland');
    }
  }
}
const names = Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`City ${i}`, String(i)]));
const full = createCitySlots(names);
assert.equal(full.encode(Object.keys(names).join(', '), 'user').filter(part => part.slot).length, 8);
assert.deepEqual(full.encode('City 8', 'user'), [{ text: 'City 8' }]);
