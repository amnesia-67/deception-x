const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../sunhacks/static/index.html'),'utf8');
class Element {
  constructor(){this.children=[];this.style={};this.hidden=false;this.disabled=false;this.textContent='';this.listeners={};this.className='';this.value='';}
  append(...items){this.children.push(...items);}
  prepend(...items){this.children.unshift(...items);}
  replaceChildren(...items){this.children=[...items];}
  addEventListener(kind,fn){this.listeners[kind]=fn;}
}
const elements=new Map();const get=id=>{if(!elements.has(id)){const node=new Element();node.id=id;elements.set(id,node);}return elements.get(id);};
const context={document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:()=>['demoBtn','liveBtn','stopBtn','finishAnswerBtn','prepareBtn','transcriptBtn','calibrateBtn'].map(get)},setTimeout:()=>0,clearTimeout(){},setInterval(){},AbortController,console};
vm.createContext(context);
const code=html.split('<script>')[1].split('</script>')[0].replace(' refresh(); setInterval(refresh,1000);','');
vm.runInContext(code,context);
const offline={mode:'stopped',config:{sample_fps:16,frames_per_window:64,capture_fps:30,hop_seconds:4},runtime:{mode:'remote_fast',status:'not_prepared',remote:{status:'offline_by_user',host:'192.168.0.224',port:5002,reason:'Teammate service is off while replacement models train.',encoder_verified:false},live_ready:false},deployment:{artifacts_status:'awaiting_retrained_weights',artifact_reason:'Replacement trained weights are pending.'}};
context.render(offline);
assert.equal(get('runtimePanel').hidden,false);
assert.equal(get('runtimeComponents').children.length,4);
assert.equal(get('liveBtn').disabled,true);
assert.equal(get('prepareBtn').disabled,false);
assert.equal(get('remoteStatus').textContent,'Offline · retraining');
assert.equal(get('remoteAddress').textContent,'192.168.0.224:5002');
assert.equal(get('fusionStatus').textContent,'Replacement pending');
assert.equal(get('fusionRawScore').textContent,'—');

assert.ok(get('prepareBtn').listeners.click);
const stale=structuredClone(offline);stale.latest={branches:{fusion:{status:'synthetic_trained',raw_score:.99,modality_gates:{visual:.5,audio:.3,text:.2},feature_distances:{D_va:.2,D_vl:.2,D_al:.2}}}};
context.render(stale);assert.equal(get('fusionRawScore').textContent,'—');assert.equal(get('fusionGateVisual').textContent,'—');assert.equal(get('fusionHeadTime').textContent,'Not measured');
const prepared=structuredClone(offline);prepared.runtime.status='partial';prepared.runtime.local_weights_loaded=true;prepared.runtime.kernels_warmed=false;prepared.runtime.components={asr:{status:'ready'},dasheng:{status:'ready'},text:{status:'ready'}};
context.render(prepared);assert.equal(get('liveBtn').disabled,true);assert.match(get('runtimeLocalNote').textContent,/first forward pass/);assert.equal(get('prepareBtn').textContent,'Check local readiness');
const preparing=structuredClone(offline);preparing.runtime.status='preparing';preparing.runtime.components={asr:{status:'loading'}};
context.render(preparing);assert.equal(get('prepareBtn').disabled,true);assert.match(get('runtimeProgress').textContent,/Local transcription/);
const legacy={mode:'stopped',deployment:{},config:{}};context.render(legacy);assert.equal(get('runtimePanel').hidden,true);assert.equal(get('liveBtn').disabled,false);
const ready=structuredClone(offline);ready.runtime.live_ready=true;ready.runtime.status='ready';ready.runtime.remote.status='verified';ready.deployment.artifacts_status='verified';context.render(ready);assert.equal(get('liveBtn').disabled,false);
for(const [status,label,headLabel] of [['awaiting_retrained_weights','Replacement weights pending','Replacement pending'],['awaiting_deployment_export','Deployment export pending','Export pending'],['awaiting_remote_verification','Verification pending','Verification pending']]){
  const pending=structuredClone(stale);pending.runtime.remote={status:'unreachable',host:'172.23.36.212',port:5002,reason:'Connection timed out after 3 seconds.'};pending.deployment.artifacts_status=status;pending.deployment.artifact_reason='';pending.deployment.requested_vision='V-JEPA 2.1 ViT-L';pending.deployment.reported_vision='V-JEPA 2.1 ViT-B';
  context.render(pending);
  assert.equal(get('remoteStatus').textContent,'unreachable');assert.match(get('remoteStatus').className,/bad/);
  assert.equal(get('remoteAddress').textContent,'172.23.36.212:5002');assert.equal(get('fusionPanelStatus').textContent,label);assert.equal(get('fusionStatus').textContent,headLabel);
  assert.equal(get('fusionRawScore').textContent,'—');assert.equal(get('fusionGateVisual').textContent,'—');assert.equal(get('fusionDistanceVA').textContent,'—');assert.equal(get('liveBtn').disabled,true);
  assert.equal(get('visionContractNote').hidden,false);assert.match(get('visionContractNote').textContent,/Requested: V-JEPA 2.1 ViT-L/);assert.match(get('visionContractNote').textContent,/Reported training model: V-JEPA 2.1 ViT-B/);assert.match(get('visionContractNote').textContent,/compatibility is unverified/);
  if(status!=='awaiting_retrained_weights'){assert.doesNotMatch(get('fusionWarning').textContent,/retrain/i);assert.doesNotMatch(get('fusionOutputReason').textContent,/retrain/i);assert.doesNotMatch(get('modeText').textContent,/retrain/i);}
}

const split={mode:'stopped',config:{capture_fps:30,sample_fps:16,frames_per_window:128},runtime:{mode:'split_local_video',status:'ready',local_weights_loaded:true,live_ready:true,remote:{status:'verified',model_id:'V-JEPA 2.1 ViT-B',host:'172.23.36.212',port:5002}},deployment:{classifier_architecture:'rlt-dasheng-digest-stack-v1',artifacts_status:'verified',local_asr_configured:true,requested_vision:'V-JEPA 2.1 ViT-L',reported_vision:'V-JEPA 2.1 ViT-B',vision_choice_accepted:true},answer:{status:'idle',elapsed_seconds:0,min_seconds:5,max_seconds:30,source_frames:0}};
context.render(split);assert.equal(get('answerPanel').hidden,false);assert.equal(get('runtimePanel').hidden,false);assert.equal(get('stackPanel').hidden,false);assert.equal(get('fusionPanel').hidden,true);assert.equal(get('liveBtn').disabled,false);assert.equal(get('finishAnswerBtn').disabled,true);assert.equal(get('runtimeComponents').children.length,6);assert.ok(get('finishAnswerBtn').listeners.click);
const recording=structuredClone(split);recording.mode='live';recording.answer={...split.answer,status:'capturing',elapsed_seconds:3.2,source_frames:95};context.render(recording);assert.equal(get('finishAnswerBtn').disabled,true);assert.equal(get('liveBtn').disabled,true);assert.equal(get('demoBtn').disabled,true);assert.equal(get('stopBtn').disabled,false);assert.equal(get('stopBtn').textContent,'Stop & analyze');assert.match(get('answerFeedback').textContent,/Keep speaking/);assert.match(get('answerFrameCount').textContent,/95 source frames/);
recording.answer.elapsed_seconds=6.4;recording.answer.source_frames=192;context.render(recording);assert.equal(get('finishAnswerBtn').disabled,false);assert.equal(get('finishAnswerBtn').textContent,'Finish answer & analyze');assert.equal(get('answerElapsed').textContent,'6.4');
for(const status of ['finishing','processing']){const analyzing=structuredClone(recording);analyzing.answer.status=status;context.render(analyzing);assert.equal(get('finishAnswerBtn').disabled,true);assert.equal(get('liveBtn').disabled,true);assert.equal(get('stopBtn').disabled,false);assert.equal(get('deviceMode').textContent,status==='finishing'?'STOPPING':'CAPTURE STOPPED');}
const complete=structuredClone(recording);complete.answer.status='complete';complete.latest={mock:false,window:{id:1,duration_seconds:6.4,transport_frames:128,native_video_unique_frames:64,effective_video_input_fps:10.0},branches:{stack:{status:'trained_classifier',raw_score:.72,calibrated:false,elapsed_ms:7,weights:{audio:.4,digest:.6}},dasheng:{status:'trained_classifier',raw_score:.6},tabular:{status:'trained_classifier',raw_score:.8,elapsed_ms:2},vision:{status:'trained_classifier',model_id:'V-JEPA 2.1 ViT-B',elapsed_ms:210}},transcript:{status:'available',source:'local_asr',text:'Example technical test'}};complete.capture={camera:{observed_fps:29.8,status:'stopped'},audio:{status:'stopped'}};
context.render(complete);assert.equal(get('liveBtn').disabled,false);assert.equal(get('liveBtn').textContent,'Start another answer');assert.equal(get('finishAnswerBtn').disabled,true);assert.equal(get('stackRawScore').textContent,'0.720');assert.equal(get('stackAudioScore').textContent,'0.600');assert.equal(get('stackDigestScore').textContent,'0.800');assert.equal(get('stackWeights').children.length,2);assert.equal(get('sampleFps').textContent,'10.0');assert.equal(get('captureFps').textContent,'29.8');assert.match(get('sampleDetail').textContent,/64 native frames · 128 sent/);assert.match(get('coverageNote').textContent,/10.0 fps effective input/);assert.equal(get('fusionRowTitle').textContent,'Local ensemble');assert.match(get('capturePrivacy').textContent,/Only video frames/);
const stoppedComplete=structuredClone(complete);stoppedComplete.mode='stopped';context.render(stoppedComplete);assert.equal(get('liveBtn').disabled,false);
const staleProcessing=structuredClone(complete);staleProcessing.answer.status='processing';context.render(staleProcessing);assert.equal(get('stackRawScore').textContent,'—');assert.equal(get('stackAudioScore').textContent,'—');assert.equal(get('stackStatus').textContent,'Awaiting completed answer');
const answerError=structuredClone(recording);answerError.answer.status='error';answerError.answer.reason='Technical test failure';context.render(answerError);assert.equal(get('liveBtn').disabled,false);assert.equal(get('answerFeedback').textContent,'Technical test failure');
const blocked=structuredClone(split);blocked.runtime.live_ready=false;blocked.runtime.reason='Video service not verified';context.render(blocked);assert.equal(get('liveBtn').disabled,true);assert.match(get('modeText').textContent,/Video service not verified/);
const pendingStack=structuredClone(complete);pendingStack.deployment.artifacts_status='awaiting_remote_verification';context.render(pendingStack);assert.equal(get('stackRawScore').textContent,'—');assert.equal(get('stackWeights').children.length,0);
const mockStack=structuredClone(complete);mockStack.mode='demo';mockStack.latest.mock=true;context.render(mockStack);assert.equal(get('stackRawScore').textContent,'—');assert.match(get('stackStatus').textContent,/Mock input/);
assert.equal(get('models').children[1].children[0].children[0].children[1].textContent,'Tuned Dasheng · 5 folds');

console.log('One-answer dashboard checks passed: capture timing, finish/processing locks, restart after complete/error, stopped completion, privacy, stack provenance, effective video fps and mock/pending suppression.');

console.log('Dashboard state checks passed: baseline six cases plus all three pending artifact statuses, new unreachable endpoint, and model mismatch.');

assert.match(html,/\.session-controls\{position:sticky;top:0;z-index:20;background:var\(--bg\)/);
const introStart=html.indexOf('<section class="intro">'), introEnd=html.indexOf('</section>',introStart), controlsIndex=html.indexOf('class="controls session-controls"');assert.ok(controlsIndex>introEnd,'Sticky controls must be outside the short intro ancestor.');
const verifiedL=structuredClone(split);verifiedL.deployment.reported_vision='V-JEPA 2.1 ViT-L';context.render(verifiedL);assert.match(get('visionContractNote').textContent,/model identity and manifest checked/);assert.doesNotMatch(get('visionContractNote').textContent,/unverified/);
const unreadyVision=structuredClone(verifiedL);unreadyVision.runtime.live_ready=false;context.render(unreadyVision);assert.match(get('visionContractNote').textContent,/unverified/);
context.render(complete);assert.match(get('fusionReason').textContent,/Learned local stack/);assert.match(get('fusionReason').textContent,/89-field/);assert.match(get('fusionReason').textContent,/7.0 ms measured/);assert.match(get('tabularReason').textContent,/89-field/);assert.match(get('tabularReason').textContent,/2.0 ms measured/);assert.doesNotMatch(get('fusionReason').textContent,/Awaiting compatible/);
for(const vision of [{status:'error',elapsed_ms:210},{status:'base',elapsed_ms:210},{status:'trained_classifier',elapsed_ms:0},{status:'trained_classifier'}]){const failed=structuredClone(complete);failed.latest.branches.vision=vision;context.render(failed);assert.equal(get('sampleFps').textContent,'—');assert.match(get('coverageNote').textContent,/did not complete/);assert.doesNotMatch(get('coverageNote').textContent,/fps effective input/);}
const stoppingDevices=structuredClone(complete);stoppingDevices.capture.camera.status='running';stoppingDevices.capture.audio.status='stopping';context.render(stoppingDevices);assert.equal(get('deviceMode').textContent,'STOPPING');
const enoughTimeFewFrames=structuredClone(recording);enoughTimeFewFrames.answer.source_frames=95;context.render(enoughTimeFewFrames);assert.equal(get('finishAnswerBtn').disabled,true);assert.equal(get('stopBtn').disabled,false);assert.equal(get('stopBtn').textContent,'Stop & analyze');

const pilotCase=structuredClone(complete);pilotCase.deployment.classifier_profile='webcam_pilot';pilotCase.deployment.classifier_architecture='webcam-pilot-dasheng-v1';pilotCase.latest.branches.stack.profile='webcam_pilot';pilotCase.latest.branches.stack.threshold=.5436123431421811;
context.render(pilotCase);assert.equal(get('stackPanel').hidden,false);assert.match(get('stackPanelTitle').textContent,/webcam-pilot audio/);assert.match(get('stackExplanation').textContent,/Only the webcam-pilot Dasheng/);assert.match(get('fusionReason').textContent,/exported calibration and cutoff/);console.log('Pilot profile UI passed: selected audio model and independent diagnostics are explicitly identified.');

const manualPilot=structuredClone(complete);
manualPilot.deployment.classifier_profile='webcam_pilot';
manualPilot.deployment.pilot_blend_weights={dasheng:.5,vision:.25,emotion:.25};
manualPilot.latest.branches.stack={status:'experimental_blend',profile:'webcam_pilot',manual_blend:true,raw_score:.55,weights:manualPilot.deployment.pilot_blend_weights,inputs:{dasheng:{score:.8},vision:{score:.2},emotion:{score:.4}}};
context.render(manualPilot);
assert.equal(get('stackRawScore').textContent,'0.550');
assert.equal(get('stackAudioScore').textContent,'0.800');
assert.equal(get('stackVideoScore').textContent,'0.200');
assert.equal(get('stackEmotionScore').textContent,'0.400');
assert.equal(get('stackDigestRow').hidden,true);
assert.equal(get('stackVideoRow').hidden,false);
assert.equal(get('stackWeights').children.length,3);
assert.match(get('stackWarning').textContent,/not been validated/);
assert.match(get('runtimeLocalNote').textContent,/manual blend/);
manualPilot.latest.branches.stack.status='unavailable';
context.render(manualPilot);
assert.equal(get('stackRawScore').textContent,'—');
assert.equal(get('stackWeights').children.length,3);
console.log('Manual blend UI checks passed: correct input scores, configured weights, and incomplete result suppression.');
