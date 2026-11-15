const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../sunhacks/static/index.html'),'utf8');
class Element{
 constructor(){this.children=[];this.style={};this.hidden=false;this.disabled=false;this.textContent='';this.listeners={};this.value='';}
 append(...x){this.children.push(...x);}prepend(...x){this.children.unshift(...x);}replaceChildren(...x){this.children=x;}
 addEventListener(k,fn){this.listeners[k]=fn;}pause(){this.paused=true;}load(){this.loaded=true;}removeAttribute(k){delete this[k];}
}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const saved=new Map();let now=0;
const context={document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:()=>[]},
 setTimeout:()=>0,clearTimeout(){},setInterval(){},AbortController,console,
 crypto:{randomUUID:()=> '12345678-1234-1234-1234-123456789abc'},performance:{now:()=>now},
 localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)}};
vm.createContext(context);
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
vm.runInContext(scripts[0].replace(' refresh(); setInterval(refresh,1000);',''),context);
vm.runInContext(scripts[1],context);
const idle={mode:'stopped',config:{},runtime:{live_ready:false},capture:{camera:{status:'stopped'},audio:{status:'stopped'}},
 dataset:{status:'idle',active:false,root:'/local/datasets',counts:{train:{truth:2,lie:3},validation:{truth:1,lie:0}},elapsed_seconds:0}};
context.render(idle);assert.equal(get('datasetStart').disabled,false,'Collection works without GPU/models.');
assert.match(get('datasetParticipant').value,/^p_/);assert.equal(saved.get('sunhacks-dataset-participant-v1'),get('datasetParticipant').value);
assert.match(get('datasetCounts').textContent,/train: 2 Truth \/ 3 Lie/);
const recording=structuredClone(idle);recording.dataset={...idle.dataset,status:'recording',active:true,elapsed_seconds:3};
context.render(recording);assert.equal(get('datasetTimer').textContent,'4.0');assert.equal(get('datasetParticipant').disabled,true);assert.equal(get('datasetStart').disabled,true);assert.equal(get('datasetDiscard').disabled,false);assert.equal(get('liveBtn').disabled,true);assert.equal(get('stopBtn').disabled,false);
const pending=structuredClone(recording);pending.dataset={...recording.dataset,status:'awaiting_label',elapsed_seconds:7,take_id:'one'};
context.render(pending);assert.equal(get('datasetLabelPanel').hidden,false);assert.equal(get('datasetTruth').disabled,false);assert.equal(get('datasetLie').disabled,false);assert.equal(get('datasetPreview').src,'/api/dataset/preview?take_id=one');assert.equal(get('datasetTimer').textContent,'0.0');
const complete=structuredClone(idle);complete.dataset.status='saved';context.render(complete);
assert.equal(get('datasetLabelPanel').hidden,true);assert.equal(get('datasetPreview').src,undefined);assert.equal(get('datasetPreview').paused,true);
(async()=>{
 const calls=[];context.api=async(path,body)=>{calls.push({path,body});return {};};context.refresh=async()=>{};
 get('datasetParticipant').value='p_same_person';await get('datasetStart').listeners.click();
 assert.equal(calls[0].path,'/api/dataset/start');assert.equal(calls[0].body.participant_id,'p_same_person');
 context.render(pending);get('datasetStatement').value='A manual statement';await get('datasetTruth').listeners.click();
 assert.equal(calls[1].body.label,'truth');assert.equal(calls[1].body.take_id,'one');assert.equal(calls[1].body.statement,'A manual statement');
 await get('datasetDiscard').listeners.click();assert.equal(calls[2].path,'/api/dataset/discard');
 context.render(idle);await get('datasetReconnectMic').listeners.click();assert.equal(calls[3].path,'/api/audio/reconnect');
 context.render(idle);context.api=async()=>{throw new Error('Camera unavailable');};await get('datasetStart').listeners.click();
 assert.match(get('datasetMessage').textContent,/Camera unavailable/);
 console.log('Dataset recorder UI passed: no model dependency, participant persistence, capture ownership, seven-second timer, preview, explicit labels, discard and errors.');
})().catch(e=>{console.error(e);process.exitCode=1;});
