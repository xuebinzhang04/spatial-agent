(function () {
  "use strict";
  const $=id=>document.getElementById(id),catalog=window.SPATIAL_CATALOG;
  let busy=false,configuredModel=null,active=null,available=false,readOnly=false,verified=false,observedRun=null,eventsSeen=0;
  const buttons=["online-reset","online-refresh","online-step","online-infer","online-model-step","online-connect"];
  const models=new Map(catalog.models.filter(m=>m.kind==="model"&&m.checkpoint).map(m=>[m.id,m]));
  const escape=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function modelChoices(){
    const previous=$("online-model").value;
    $("online-model").innerHTML='<option value="manual">Manual API</option>'+Array.from(models.values()).map(m=>'<option value="'+escape(m.id)+'">'+escape(m.name)+(m.checkpoint?(m.downloaded===false?' · incomplete checkpoint':' · downloaded'):' · service')+'</option>').join('');
    $("online-model").value=models.has(previous)?previous:'manual';
  }
  function modelStatus(){
    const id=$("online-model").value,m=models.get(id);
    $("online-model-status").textContent=readOnly?'Observation-only connection · worker model: '+configuredModel:id==='manual'?'Manual environment actions':(m&&m.checkpoint?'Checkpoint: '+m.checkpoint+' · ':'')+(id===configuredModel?(verified?'Inference service verified':'Inference endpoint configured'):'No inference service connected for this model');
  }
  const tasks=()=>catalog.tasks.filter(t=>t.environment===$("online-environment").value&&t.online);
  const selected=()=>tasks().find(t=>t.id===$("online-task").value);
  function controls(){
    buttons.forEach(id=>$(id).disabled=busy||!available);
    const delivery=$("online-environment").value==="simworld_delivery";
    $("online-reset").disabled=busy||!available||readOnly||delivery||!selected();
    $("online-scene").disabled=$("online-environment").value!=="ai2thor";
    $("online-level").disabled=delivery;
    $("online-level").max=$("online-environment").value==="ai2thor"?"3":"5";
    if(Number($("online-level").value)>Number($("online-level").max))$("online-level").value=$("online-level").max;
    const modelReady=available&&!readOnly&&configuredModel&&$("online-model").value===configuredModel;
    const match=active&&selected()&&active.environment===$("online-environment").value&&active.task===selected().online.task;
    $("online-step").disabled=busy||!available||readOnly||!match;
    $("online-infer").disabled=busy||!modelReady||!match;
    $("online-model-step").disabled=busy||!modelReady||!match;
    $("online-connect").disabled=busy||!available||readOnly||$("online-model").value==='manual';
    $("online-model-url").disabled=readOnly;
    $("online-stop").disabled=!available;
    $("online-stop").textContent=readOnly?'Stop viewer':'Stop session';
    modelStatus();
  }
  function describe(){
    const task=selected(),env=catalog.environments.find(e=>e.id===$("online-environment").value);
    $("online-description").textContent=task?env.name+" + "+task.label+". Objective: "+task.research.objective+(env.id==="simworld_delivery"?" Read the allocated session; relaunch it to reset or change its city.":""):"No implemented online task for this simulator.";
    const examples={shape3d:'{"action":"CAMERA_RIGHT"}',ai2thor:'{"action":"RotateRight"}',simworld_delivery:'VIEW_ORDERS()'};
    $("online-action").value=examples[$("online-environment").value]||"";
    controls();
  }
  function taskChoices(){const values=tasks();$("online-task").innerHTML=values.map(t=>'<option value="'+t.id+'">'+t.label+'</option>').join("");$("online-task").value=values.length?values[0].id:"";describe();}
  $("online-environment").innerHTML=catalog.environments.filter(e=>catalog.tasks.some(t=>t.environment===e.id&&t.online)).map(e=>'<option value="'+e.id+'">'+e.name+'</option>').join("");
  $("online-environment").value="shape3d";taskChoices();
  $("online-environment").addEventListener("change",taskChoices);
  $("online-task").addEventListener("change",describe);
  modelChoices();
  controls();
  $("online-model").addEventListener("change",controls);
  async function request(path,value){
    const response=await fetch(path,value===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(value)});
    const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;
  }
  function display(state,reset,response,scroll=true){
    active=state;
    if($("online-environment").value!==state.environment){$("online-environment").value=state.environment;taskChoices();}
    const current=tasks().find(t=>t.online.task===state.task);
    if(current&&$("online-task").value!==current.id){$("online-task").value=current.id;describe();}
    if(state.events){
      if(observedRun!==state.run_id){observedRun=state.run_id;eventsSeen=0;}
      state.events.slice(eventsSeen).forEach(event=>SpatialDemo.appendOnline(event.state,event.event==='reset',event.state.model_output));
      eventsSeen=state.events.length;
    }
    SpatialDemo.appendOnline(state,reset,response||state.model_output);
    $("online-api").textContent=typeof state.actions==="string"?state.actions:JSON.stringify(state.actions,null,2);
    $("online-status").textContent=(readOnly?'Worker '+state.run_id+' · '+state.run_status+' · ':'')+state.environment+" · environment step "+state.step+" · "+(state.feedback||"Observation received");
    if(scroll)$("explorer").scrollIntoView({block:"start"});
  }
  async function operate(operation){
    if(busy)return;busy=true;controls();$("online-status").textContent="Waiting for the environment/model operation to complete…";
    try{await operation();}catch(error){$("online-status").textContent=error.message;}finally{busy=false;controls();}
  }
  $("online-reset").addEventListener("click",()=>operate(async()=>{
    const task=selected();if(!task)throw new Error("Select an implemented task");
    display(await request("/api/reset",{environment:task.online.environment,task:task.online.task,seed:Number($("online-seed").value),difficulty:Number($("online-level").value),scene:$("online-scene").value}),true);
  }));
  $("online-refresh").addEventListener("click",()=>operate(async()=>display(await request("/api/state"),false)));
  $("online-connect").addEventListener("click",()=>operate(async()=>{
    const result=await request('/api/model',{model:$("online-model").value,url:$("online-model-url").value});
    configuredModel=result.model;verified=result.model_verified;$("online-status").textContent='Connected to '+configuredModel;
  }));
  $("online-step").addEventListener("click",()=>operate(async()=>{const raw=$("online-action").value.trim();display(await request("/api/action",{action:raw.startsWith("{")?JSON.parse(raw):raw,expected_environment:$("online-environment").value,expected_task:selected().online.task}),false);}));
  async function infer(execute){
    const result=await request("/api/infer",{execute,model:$("online-model").value,expected_environment:$("online-environment").value,expected_task:selected().online.task});
    $("online-model-response").textContent=result.model+"\n"+result.response;
    $("online-action").value=typeof result.action==="string"?result.action:JSON.stringify(result.action);
    if(result.state)display(result.state,false,result.response);else $("online-status").textContent="Model proposed an action; it has not been executed.";
  }
  $("online-infer").addEventListener("click",()=>operate(()=>infer(false)));
  $("online-model-step").addEventListener("click",()=>operate(()=>infer(true)));
  $("online-stop").addEventListener("click",async()=>{try{$("online-status").textContent=(await request("/api/stop",{})).status;}catch(error){$("online-status").textContent=error.message;}});
  setInterval(async()=>{
    if(!available||busy||!$("online-follow").checked)return;
    busy=true;
    try{display(await request('/api/state'),false,null,false);}catch(error){$("online-status").textContent=error.message;}finally{busy=false;controls();}
  },2000);
  if(window.location.protocol!=="file:")request("/api/capabilities").then(async result=>{
    available=true;readOnly=Boolean(result.read_only);verified=Boolean(result.model_verified);
    (result.models||[]).forEach(m=>models.set(m.id,m));
    configuredModel=result.model;
    if(configuredModel&&!models.has(configuredModel))models.set(configuredModel,{id:configuredModel,name:configuredModel});
    modelChoices();
    if(configuredModel)$("online-model").value=configuredModel;
    if(result.model_url)$("online-model-url").value=result.model_url;
    if(result.environment){$("online-environment").value=result.environment;taskChoices();const task=tasks().find(t=>t.online.task===result.task);if(task){$("online-task").value=task.id;describe();}}
    $("online-status").textContent=readOnly?'Observing independent inference worker':'Online API ready · configured model: '+(configuredModel||'none');controls();
    if(readOnly){$("online-follow").checked=true;if(result.environment)display(await request('/api/state'),false,null,false);}
  }).catch(error=>{$("online-status").textContent="Online API unavailable: "+error.message;});
}());
