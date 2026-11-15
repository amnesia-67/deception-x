/* Mochi reads existing UI state only. No network, sensors or inference. */
(()=>{'use strict';
 const root=document.getElementById('mochiCompanion');if(!root)return;
 const message=document.getElementById('mochiMessage'),phase=document.getElementById('mochiPhase'),pet=document.getElementById('mochiPet'),help=document.getElementById('mochiHelp'),toggle=document.getElementById('mochiHelpButton');
 let current=null,petUntil=0,petCount=0,petTimer=null;
 const active=s=>['starting','running','stopping'].some(status=>Object.values(s.capture||{}).some(x=>x.status===status));
 const seconds=v=>typeof v==='number'&&Number.isFinite(v)?Math.max(0,Math.ceil(v)):null;
 function describe(s){
  if(s.kind==='dashboard'){
   const d=s.dataset||{},remaining=seconds((d.duration_seconds||7)-(d.elapsed_seconds||0));
   if(d.status==='awaiting_label')return ['label','Recording stopped. Was that a truth or a lie? Choose its label to save it.','YOUR LABEL, YOUR CALL'];
   if(d.active&&['recording','capturing'].includes(d.status))return ['recording',`I’m listening! ${remaining??7} seconds left. Your clip stops automatically.`,'CAMERA + MIC ON'];
   if(d.active)return ['processing','Getting your clip ready. Wait for the label buttons before the next take.','SAVING YOUR MOMENT'];
   if(d.status==='error')return ['error','That recording hit a snag. Check the recorder message; reconnect the mic if needed.','LET’S FIX IT'];
   if(active(s))return ['recording','Camera and microphone are active. Use Stop when you want to finish.','CAPTURE ACTIVE'];
   if(['processing','finishing'].includes(s.answer?.status))return ['processing','Recording has stopped. The models are working on that answer.','MODELS WORKING'];
   const counts=Object.values(d.counts||{}).reduce((n,v)=>n+(v.truth||0)+(v.lie||0),0);
   return ['idle',counts?`${counts} labeled clips tucked away! Keep your participant code, record seven seconds, then pick Truth or Lie.`:'Ready to collect a clip? Keep your participant code, record seven seconds, then pick Truth or Lie.','YOUR RECORDING BUDDY'];
  }
  const stage=s.stage||'idle',left=seconds(s.remaining_seconds);
  if(stage==='recording')return ['recording',`You’ve got this. ${left??7} seconds left—keep speaking naturally!`,'CAMERA + MIC ON'];
  if(stage==='warmup')return ['recording',s.prompt?.body||'Two simple truths, then one harmless lie. Just a little practice.','PRACTICE · '+(left??15)+'s LEFT'];
  if(['starting','answer_starting'].includes(stage))return ['processing','Getting your camera and mic ready. Speak when the countdown begins.','GETTING READY'];
  if(['processing','warmup_review'].includes(stage))return ['processing','All captured! Camera and microphone are stopped. I’ll keep you company while we wait.','RECORDING STOPPED'];
  if(stage==='result')return ['result','You did it! Want another round? Pick a fresh statement and keep your choice secret.','ROUND COMPLETE'];
  if(stage==='error')return ['error','A little hiccup. Check the message beside me before trying again.','HERE TO HELP'];
  if(stage==='cancelled')return ['idle','No rush. Your session is stopped. We can try again whenever you’re ready.','TAKE YOUR TIME'];
  if(s.runtime_blocked===true)return ['error','The video model isn’t ready. Check its connection in the model dashboard.','CHECK CONNECTION'];
  if(s.can_prepare===true&&s.preparing!==true&&!(s.model_ready===true||s.runtime?.live_ready===true))return ['error','The models need a connection check. Use the button beside me before starting.','CHECK CONNECTION'];
  if(!(s.model_ready===true||s.runtime?.live_ready===true))return ['processing','The models are getting ready. I’m warming up my listening ears.','PREPARING MODELS'];
  return ['idle','Hi, I’m Mochi. One statement, seven seconds. I’ll help you keep track.','YOUR SEVEN-SECOND SIDEKICK'];
 }
 function updatePulse(s){
  const panel=document.getElementById('mochiPulse');if(!panel)return;
  const hr=s.heart_rate||{},live=hr.status==='tracking'&&Number.isFinite(hr.bpm),warm=['acquiring','no_face'].includes(hr.status);
  const saved=!live&&!warm&&Number.isFinite(hr.recorded_reading?.bpm)?hr.recorded_reading:null,reading=live?hr:saved;
  panel.dataset.live=String(live);panel.dataset.recorded=String(Boolean(saved));document.getElementById('mochiPulseBpm').textContent=reading?String(Math.round(reading.bpm)):'—';
  document.getElementById('mochiPulseStatus').textContent=saved?'Recorded BPM':live?(hr.quality_label==='good'?'Good signal':hr.quality_label==='fair'?'Fair signal':'Weak signal'):hr.status==='no_face'?'Find your face':warm?'Acquiring':'Camera off';
  const pulse=reading&&Array.isArray(reading.pulse)?reading.pulse.filter(Number.isFinite):[];
  document.getElementById('mochiPulseLine').setAttribute('points',pulse.length>1?pulse.map((v,i)=>(180*i/(pulse.length-1)).toFixed(1)+','+(14-10*Math.max(-1,Math.min(1,v))).toFixed(1)).join(' '):'');
  document.getElementById('mochiPulseNote').textContent=saved?'Last valid reading from this recording · '+(saved.quality_label||'estimated')+' signal · camera is off':live?'On-device camera estimate · display only · separate from the verdict':warm?'Hold still in even light. Gathering a pulse signal…':'On-device estimate · appears during recording · separate from the verdict';

 }
 function update(s){current=s;updatePulse(s);const [m,text,status]=describe(s);root.dataset.mood=m;phase.textContent=status;if(Date.now()>=petUntil)message.textContent=text;help.textContent=s.kind==='dashboard'?'Keep one code per person. Click Add new data, speak for seven seconds, then label your own statement Truth or Lie. Recording alone never saves a label.':'Choose a truth or a harmless lie and keep your choice secret. Click Start, wait for the countdown, and speak naturally for seven seconds. Practice is optional. The prediction can be wrong. Petting me never starts a recording.';}
 pet.addEventListener('click',()=>{petCount++;petUntil=Date.now()+2800;root.dataset.petted='false';void root.offsetWidth;root.dataset.petted='true';message.textContent=['A tiny scritch! A very big thank-you. ♡','Soft ears. Serious listening. You’ve got this!','Another scritch? Best. Hackathon. Ever.'][((petCount-1)%3)];clearTimeout(petTimer);petTimer=setTimeout(()=>{root.dataset.petted='false';if(current)update(current);},2850);});
 toggle.addEventListener('click',()=>{help.hidden=!help.hidden;toggle.setAttribute('aria-expanded',String(!help.hidden));toggle.textContent=help.hidden?'How it works':'Got it';});
 globalThis.Mochi={update,describe};update({kind:root.dataset.context||'showcase'});
})();
