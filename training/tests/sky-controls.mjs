import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const classes = new Set();
const doc = { hidden: false,
  addEventListener: (_, fn) => doc.visibility = fn,
  body: { classList: { toggle: (key, on) => on ? classes.add(key) : classes.delete(key) } },
};
vm.runInNewContext(fs.readFileSync(new URL('../../web/dist/sky.js', import.meta.url), 'utf8'), { document: doc });
assert.equal(classes.has('sky-idle'), false);
doc.hidden = true; doc.visibility();
assert.equal(classes.has('sky-idle'), true);
doc.hidden = false; doc.visibility();
assert.equal(classes.has('sky-idle'), false);
console.log('Passed: hidden-tab animation suspension and resume without a pause control.');
