import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../../web/dist/sky.js',import.meta.url),'utf8');
function setup(reduced=false) {
  const classes = new Set();
  const attributes = {};
  const button = { hidden:true, addEventListener:(_,fn)=>button.click=fn, setAttribute:(key,value)=>attributes[key]=value };
  const label = { textContent:'' };
  const motion = { matches:reduced, addEventListener:(_,fn)=>motion.change=fn };
  const doc = { hidden:false, getElementById:id=>id==='sky-toggle'?button:label,
    addEventListener:(_,fn)=>doc.visibility=fn,
    body:{classList:{toggle:(key,on)=>on?classes.add(key):classes.delete(key)}} };
  vm.runInNewContext(source,{ document:doc, window:{matchMedia:()=>motion} });
  return { classes,attributes,button,label,motion,doc };
}
const normal=setup();
assert.equal(normal.button.hidden,false);
assert.equal(normal.attributes['aria-label'],'Pause birds');
normal.button.click();
assert.ok(normal.classes.has('sky-paused'));
assert.equal(normal.attributes['aria-pressed'],'true');
assert.equal(normal.label.textContent,'Play birds');
normal.doc.hidden=true;normal.doc.visibility();
normal.doc.hidden=false;normal.doc.visibility();
assert.ok(normal.classes.has('sky-paused'),'Returning to the tab must preserve a user pause');
normal.button.click();
assert.equal(normal.classes.has('sky-paused'),false);
normal.doc.hidden=true;normal.doc.visibility();
assert.ok(normal.classes.has('sky-idle'));
normal.doc.hidden=false;normal.doc.visibility();
assert.equal(normal.classes.has('sky-idle'),false);
const still=setup(true);
assert.equal(still.button.hidden,true);
still.motion.matches=false;still.motion.change();
assert.equal(still.button.hidden,false);
console.log('Passed: pause/resume, accessible labels, persistent manual pause, tab suspension, and reduced-motion control.');
