const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../sunhacks/static/index.html'),'utf8');
class Element {
  constructor(){this.children=[];this.style={};this.hidden=false;this.disabled=false;this.textContent='';this.listeners={};this.className='';this.value='';this.attributes={};}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=[...items];}
  addEventListener(kind,fn){this.listeners[kind]=fn;}
  setAttribute(name,value){this.attributes[name]=String(value);}
}
const elements=new Map();const get=id=>{if(!elements.has(id)){const node=new Element();node.id=id;elements.set(id,node);}return elements.get(id);};
const context={document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:()=>[]},setTimeout:()=>0,clearTimeout(){},setInterval(){},AbortController,console};
vm.createContext(context);
const code=html.split('<script>')[1].split('</script>')[0].replace(' refresh(); setInterval(refresh,1000);','');
vm.runInContext(code,context);
for(const id of ['heartRatePanel','heartRateBpm','heartRateQuality','heartRatePulse','heartRateDetail','heartRateIcon'])assert.ok(html.includes('id="'+id+'"'),id);
const idle={mode:'stopped',config:{},deployment:{},capture:{camera:{status:'stopped'},audio:{status:'stopped'}},heart_rate:{status:'idle',display_only:true,bpm:null,pulse:[]}};
context.render(idle);assert.equal(get('heartRatePanel').hidden,false);
const acquiring=structuredClone(idle);acquiring.mode='live';acquiring.capture.camera={status:'running',observed_fps:30};acquiring.heart_rate={status:'acquiring',display_only:true,bpm:null,progress:.42,method:'facephys',pulse:[],model:{status:'ready'}};
context.render(acquiring);assert.equal(get('heartRatePanel').hidden,false);assert.equal(get('heartRateBpm').textContent,'—');assert.equal(get('heartRateQuality').textContent,'Acquiring 42%');assert.match(get('heartRateDetail').textContent,/Hold still/);assert.doesNotMatch(get('heartRateIcon').className,/beat/);
const tracking=structuredClone(acquiring);tracking.heart_rate={status:'tracking',display_only:true,bpm:72.4,quality:.81,quality_label:'good',method:'facephys',window_seconds:10,pulse:[0,.5,1,.5,0,-.5,-1,-.5,0],progress:1};
context.render(tracking);assert.equal(get('heartRateBpm').textContent,'72');assert.equal(get('heartRateQuality').textContent,'Good signal');assert.match(get('heartRateQuality').className,/good/);assert.match(get('heartRateIcon').className,/beat/);assert.equal(get('heartRatePulse').attributes.points.split(' ').length,9);assert.match(get('heartRateDetail').textContent,/FacePhys on-device.*display only/);
const weak=structuredClone(tracking);weak.heart_rate.quality_label='poor';weak.heart_rate.method='pos';context.render(weak);assert.equal(get('heartRateQuality').textContent,'Weak signal');assert.match(get('heartRateValue').className,/dim/);assert.match(get('heartRateDetail').textContent,/POS fallback/);
const noFace=structuredClone(acquiring);noFace.heart_rate.status='no_face';context.render(noFace);assert.equal(get('heartRateQuality').textContent,'No face');
const demo=structuredClone(tracking);demo.mode='demo';demo.capture.camera={status:'mock'};demo.heart_rate={status:'idle',pulse:[]};context.render(demo);assert.equal(get('heartRatePanel').hidden,false);
const stopped=structuredClone(tracking);stopped.mode='stopped';stopped.capture.camera={status:'stopped'};stopped.heart_rate={status:'idle',pulse:[]};context.render(stopped);assert.equal(get('heartRatePanel').hidden,false);
const legacy={mode:'stopped',config:{},deployment:{}};context.render(legacy);assert.equal(get('heartRatePanel').hidden,true);
console.log('Heart-rate widget checks passed: camera-off placeholder, acquiring progress, BPM/quality/sparkline when tracking, POS fallback label, no-face state.');

const recorded=structuredClone(idle);recorded.heart_rate.recorded_reading={bpm:73.2,quality:.8,quality_label:'good',method:'facephys',pulse:[0,1,0,-1,0],window_seconds:6.8};
context.render(recorded);assert.equal(get('heartRateBpm').textContent,'73');assert.equal(get('heartRateQuality').textContent,'Recorded BPM');assert.match(get('heartRateDetail').textContent,/Last valid reading.*camera off/);assert.doesNotMatch(get('heartRateIcon').className,/beat/);
context.render(idle);assert.equal(get('heartRateBpm').textContent,'—');
