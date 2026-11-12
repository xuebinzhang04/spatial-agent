(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const escape = value => String(value == null ? "" : value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const notation = value => String(value == null ? "" : value).split(/(`[^`]+`)/g).map(part => part.startsWith("`") && part.endsWith("`") ? "<code>" + escape(part.slice(1,-1)) + "</code>" : escape(part)).join("");
  const title = value => String(value || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  let catalog;
  const state = {task: null, agent: null, episode: null, step: 0, environment: "all", timer: null, view: null, processSample: null, processMode: "continuous"};
  let onlineEpisode = null;
  let processLoaded = null;
  const task = () => catalog.tasks.find(t => t.id === state.task);
  const episode = () => catalog.episodes.find(e => e.id === state.episode);
  const model = () => catalog.models.find(m => m.id === state.agent);
  const taskEpisodes = () => catalog.episodes.filter(e => e.task === state.task);
  const agentEpisodes = () => taskEpisodes().filter(e => e.agent === state.agent).sort((a,b) => Number(Boolean(b.process))-Number(Boolean(a.process)) || a.difficulty - b.difficulty || a.id.localeCompare(b.id));
  const text = (id, value) => { $(id).textContent = value; };

  function validate(data) {
    if (!data || data.schema !== "spatial-agent-demo/1" || !Array.isArray(data.tasks) || !Array.isArray(data.episodes) || !Array.isArray(data.models) || !Array.isArray(data.environments)) throw new Error("Unsupported episode catalog");
    const taskIds = new Set(data.tasks.map(t => t.id));
    const modelIds = new Set(data.models.map(m => m.id));
    const episodeIds = new Set();
    for (const e of data.episodes) {
      if (episodeIds.has(e.id) || !taskIds.has(e.task) || !modelIds.has(e.agent) || !Array.isArray(e.frames) || !e.frames.length) throw new Error("Invalid episode binding");
      if (!e.result || !e.provenance || typeof e.provenance.privileged_policy !== "boolean") throw new Error("Missing episode provenance");
      if ((e.agent === "oracle") !== e.provenance.privileged_policy) throw new Error("Agent identity contradicts policy provenance");
      e.frames.forEach((f,i) => { if (f.step !== i || typeof f.image !== "string" || !f.observation) throw new Error("Invalid episode frame"); });
      episodeIds.add(e.id);
    }
  }

  function pause() {
    ["process-fpv","process-global","process-local"].forEach(id=>$(id).pause());
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null;
    $("play-button").classList.remove("playing");
    $("play-button").setAttribute("aria-label", "Play recorded trajectory");
  }

  function play() {
    const e = episode();
    if (!e) return;
    if(e.process&&state.processMode==="continuous"){
      const video=$("process-fpv");if(video.paused){const promise=video.play();if(promise)promise.catch(()=>{});}else video.pause();return;
    }
    if (state.timer !== null) { pause(); return; }
    if (state.step === e.frames.length - 1) seek(0);
    state.timer = setInterval(() => {
      if (state.step >= e.frames.length - 1) { pause(); return; }
      seek(state.step + 1, false);
      if (state.step >= e.frames.length - 1) pause();
    }, Number($("speed-select").value) || 750);
    $("play-button").classList.add("playing");
    $("play-button").setAttribute("aria-label", "Pause recorded trajectory");
  }

  function chooseTask(id) {
    const selected = catalog.tasks.find(t => t.id === id);
    if (!selected) return false;
    pause();
    state.task = id;
    if(state.environment!=="all"&&state.environment!==selected.environment)state.environment=selected.environment;
    state.step = 0; state.processSample=null; state.processMode="continuous";
    const eps = taskEpisodes();
    const previousAgent = state.agent;
    const processEpisode = eps.find(e => e.process && e.agent === state.agent) || eps.find(e => e.process);
    if (processEpisode) state.agent = processEpisode.agent;
    else if (!eps.some(e => e.agent === state.agent)) state.agent = eps.length ? eps[0].agent : null;
    state.episode = agentEpisodes().length ? agentEpisodes()[0].id : null;
    text("selection-notice", previousAgent && previousAgent !== state.agent && state.agent === "oracle" ? "Oracle reference selected: this task has no imported rollout for the previous model." : "");
    renderSelection();
    renderFrame();
    return true;
  }

  function chooseEnvironment(id) {
    if (id !== "all" && !catalog.environments.some(e => e.id === id)) return false;
    state.environment = id;
    const visible = catalog.tasks.filter(t => id === "all" || t.environment === id);
    const preferred = visible.find(t => catalog.episodes.some(e => e.task === t.id && e.process)) || visible[0];
    if (visible.length && !visible.some(t => t.id === state.task)) chooseTask(preferred.id);
    else renderSelection();
    return true;
  }

  function chooseAgent(id) {
    if (!taskEpisodes().some(e => e.agent === id)) return false;
    pause();
    state.agent = id;
    state.episode = agentEpisodes()[0].id;
    state.step = 0; state.processSample=null; state.processMode="continuous";
    text("selection-notice", "");
    renderSelection(); renderFrame();
    return true;
  }

  function chooseEpisode(id) {
    const e = catalog.episodes.find(item => item.id === id && item.task === state.task && item.agent === state.agent);
    if (!e) return false;
    pause(); state.episode = id; state.step = 0; state.processSample=null; state.processMode="continuous";
    $("episode-select").value = id;
    renderFrame();
    return true;
  }

  function renderResearch() {
    const t = task(), r = t && t.research || {};
    const env = catalog.environments.find(e => t && e.id === t.environment);
    $("delivery-map-tools").hidden=!t || t.environment!=="simworld_delivery";
    for (const field of ["definition", "objective", "actions", "verification", "difficulty", "limits"]) {
      const value = r[field] || "Not documented for this task.";
      text("demo-" + field, value);
      $("demo-" + field).innerHTML = notation(value);
    }
    $("demo-variables").innerHTML = (env && env.variables || []).concat(r.factors || []).map(row => "<tr>" + row.map(cell => "<td>" + notation(cell) + "</td>").join("") + "</tr>").join("");
    text("demo-simulator", env ? env.definition + "\nRuntime: " + env.runtime + "\nRevision: " + env.revision + "\nEvidence: " + env.evidence : "No simulator build documented.");
  }

  function renderSelection() {
    const current = task();
    renderResearch();
    $("environment-select").value = state.environment;
    const visible = catalog.tasks.filter(t => state.environment === "all" || t.environment === state.environment);
    $("task-select").innerHTML = visible.map(t => '<option value="'+escape(t.id)+'">'+escape((t.label||title(t.id))+(t.episode_ids.length?"":" (pending)"))+'</option>').join("");
    $("task-select").value=state.task;
    $("agent-select").innerHTML = catalog.models.filter(m => taskEpisodes().some(e => e.agent === m.id)).map(m => '<option value="' + escape(m.id) + '">' + escape(m.name) + '</option>').join("");
    if (state.agent) $("agent-select").value = state.agent;
    $("agent-select").disabled = !taskEpisodes().length;
    $("episode-select").innerHTML = agentEpisodes().map(e => '<option value="' + escape(e.id) + '">' + escape(e.scene + (e.difficulty>0?" · L"+e.difficulty:"") + " · seed " + e.seed + (e.provenance.collection?" · "+e.provenance.collection:"")+(e.process?" · full process · "+e.id:current.environment==="simworld_delivery"?" · "+e.frames.length+" snapshots only":"")) + '</option>').join("");
    $("episode-select").value = state.episode || "";
    $("episode-select").disabled = !episode();
    if (current) {
      text("task-title", current.title);
    }
    const knownModels = catalog.models.filter(m => m.kind === "model" && m.checkpoint);
    text("model-caption", model() && model().checkpoint ? "Model: " + model().name + " · " + model().checkpoint : "Local checkpoints: " + (knownModels.map(m => m.name).join(", ") || "none catalogued"));
  }

  function describeAction(action) {
    if (typeof action === "string") return action;
    const name = action && (action.action || action.op);
    const labels = {
      MoveAhead:"Move forward", MoveBack:"Move backward", RotateLeft:"Turn left", RotateRight:"Turn right",
      LookUp:"Look up", LookDown:"Look down", OpenObject:"Open container", CloseObject:"Close container",
      PickupObject:"Pick up object", PutObject:"Place carried object", PLACE:"Place at selected slot",
      MEMORIZE:"Commit memory", MARK:"Mark landmark", FINISH:"Submit scene state",
      SUBMIT:"Submit change report", SUBMIT_SHAPE:"Submit shape selection", SUBMIT_POSE:"Submit pose alignment", WAIT:"Wait"
    };
    return name ? labels[name] || title(name) : "Initial state";
  }

  function renderProcess(e) {
    const available=Boolean(e&&e.process);
    if(!available)processLoaded=null;
    $("process-controls").hidden=!available;
    if(available&&processLoaded!==e.id){
      processLoaded=e.id;state.processSample=null;state.processMode="continuous";
      [["process-fpv","First-person RGB"],["process-global","Global map"],["process-local","Local map"]].forEach(pair=>{
        const video=$(pair[0]);video.src=e.process.streams.find(v=>v.name===pair[1]).url;video.load();
      });
    }
    const full=available&&state.processMode==="continuous";
    $("process-mode").value=state.processMode;
    $("delivery-process").hidden=!full;
    ["viewport","decision-playback"].forEach(id=>$(id).hidden=full);
    $("decision-view-controls").hidden=full||!e||(e.frames[state.step].views||[]).length<2;
    if(!full)["process-fpv","process-global","process-local"].forEach(id=>$(id).pause());
    const samples=available?e.process.samples:[],duration=available?samples[samples.length-1].time_s-samples[0].time_s:0;
    text("process-caption",available?duration.toFixed(0)+" s · "+samples.length+" samples · "+(e.frames.length-1)+" decisions":"");
  }

  function syncProcess() {
    const e=episode();if(!e||!e.process||state.processMode!=="continuous")return;
    const video=$("process-fpv"),time=video.currentTime+e.process.time_offset_s;
    const samples=e.process.samples;let index=0;
    while(index+1<samples.length&&samples[index+1].time_s<=time)index++;
    const sample=samples[index];
    if(state.processSample!==sample){state.processSample=sample;state.step=Math.min(e.frames.length-1,sample.action_index);renderFrame();}
    if(window.CityDemo&&window.CityDemo.setRecordedState)window.CityDemo.setRecordedState(e.scene,sample.state,samples.slice(0,index+1).map(s=>s.state.position_cm));
    text("process-time","Case time "+time.toFixed(1)+" s · observation "+(index+1)+" / "+samples.length+" · decision "+sample.action_index);
    ["process-global","process-local"].forEach(id=>{const map=$(id);if(Math.abs(map.currentTime-video.currentTime)>.6)map.currentTime=video.currentTime;});
  }

  function renderFrame() {
    const e = episode(), t = task();
    const empty = !e;
    $("empty-state").hidden = !empty;
    $("current-wrap").hidden = empty;
    for (const id of ["previous-button","next-button","play-button","timeline","end-button"]) $(id).disabled = empty;
    $("objective-list").innerHTML = (t ? t.objectives : []).map(s => '<li class="' + (e && state.step === e.frames.length-1 && e.result.success && (!state.processSample||state.processSample.state.phase==="complete") ? "complete" : "") + '">' + escape(s) + '</li>').join("");
    if (empty) {
      renderProcess(null);
      pause();
      for (const id of ["scene-label","step-counter","state-phase","state-budget","state-agent","state-reward","state-memory","action-index"]) text(id,"—");
      text("action-label","No recorded action");
      text("episode-status","No completed recording imported"); $("episode-status").className="episode-status";
      text("policy-note","Catalog entry only; no native validation result is implied by this view.");
      text("step-record","No recorded transition is available for this task.");
      $("view-select").innerHTML="";
      $("timeline").max="0"; $("timeline").value="0";
      $("model-output-details").hidden=true;
      text("model-input","No model input recorded for this task.");
      $("model-exchange").hidden=true;
      text("provenance-content", "No recording available.");
      text("state-resources","");
      $("case-difficulty").innerHTML="";
      text("observation-phase","");
      return;
    }
    renderProcess(e);
    const f = e.frames[state.step], before = e.frames[Math.max(0,state.step-1)], obs = state.processSample?state.processSample.state:f.observation;
    const design=e.difficulty_design || {level:e.difficulty,seed:e.seed,scene:e.scene,action_budget:e.budget,empirically_calibrated:false};
    $("case-difficulty").innerHTML="<p>Selected case parameters</p><pre>"+escape(JSON.stringify(design,null,2))+"</pre>";

    text("step-record",JSON.stringify({step:f.step,environment_step:f.environment_step===undefined?f.step:f.environment_step,action:f.action,reward:f.reward,terminated:state.processSample?obs.phase==="complete":f.terminated,truncated:state.processSample?false:f.truncated,observation:obs},null,2));
    const views=f.views || [{name:"Agent observation",image:f.image}];
    if (!views.some(v=>v.name===state.view)) state.view=views.find(v=>v.name==="First-person RGB")?"First-person RGB":views[0].name;
    $("view-select").innerHTML=views.map(v=>'<option value="'+escape(v.name)+'">'+escape(v.name)+'</option>').join("");
    $("view-select").value=state.view;
    $("observation-image").src=views.find(v=>v.name===state.view).image;
    $("observation-image").alt=t.title+", step "+state.step;
    const phases=t.research&&t.research.phase_labels||{};
    text("observation-phase",(e.presentation_phase_notes||{})[obs.phase]||phases[obs.phase]||title(obs.phase||"interaction"));
    const phase=title(obs.phase || "interaction");
    text("state-phase",phase);
    text("scene-label",((catalog.environments.find(item=>item.id===t.environment)||{}).name||t.environment)+" · "+e.scene+(e.difficulty>0?" · L"+e.difficulty:""));
    text("step-counter",String(state.step).padStart(2,"0")+" / "+(e.frames.length-1)+" steps");
    const resources={};for(const key of ["position_cm","sim_time_s","energy_pct","battery_pct","transport","orders","bag"])if(obs[key]!==undefined)resources[key]=obs[key];
    if(t.environment==="simworld_delivery"&&!state.processSample&&window.CityDemo&&window.CityDemo.setRecordedState)window.CityDemo.setRecordedState(e.scene,resources,[]);

    const resourceSummary=[];
    if(obs.battery_pct!==undefined)resourceSummary.push("Battery "+Number(obs.battery_pct).toFixed(0)+"%");
    if(obs.energy_pct!==undefined)resourceSummary.push("Energy "+Number(obs.energy_pct).toFixed(0)+"%");
    if(obs.sim_time_s!==undefined)resourceSummary.push("Time "+Number(obs.sim_time_s).toFixed(0)+" s");
    text("state-resources",resourceSummary.join(" · "));
    $("state-resources").hidden=!resourceSummary.length;

    text("state-budget",Math.max(0,e.budget-(f.environment_step===undefined?state.step:f.environment_step))+" / "+e.budget);
    text("state-agent",f.agent || (model()?model().name:e.agent));
    text("state-reward",(f.total_reward===undefined?e.frames.slice(0,state.step+1).reduce((sum,r)=>sum+r.reward,0):f.total_reward).toFixed(3));
    text("state-memory",[obs.feedback ? "Feedback: " + obs.feedback : "", obs.memory ? "Memory: " + obs.memory : ""].filter(Boolean).join("\n\n"));
    $("state-memory").hidden=!obs.feedback&&!obs.memory;
    $("timeline").max=String(e.frames.length-1); $("timeline").value=String(state.step);
    $("previous-button").disabled=state.step===0; $("next-button").disabled=state.step===e.frames.length-1;
    text("action-label",describeAction(f.action));
    text("action-index","step "+state.step);
    text("provenance-content",JSON.stringify({episode:e.id,agent:e.agent,scene:e.scene,seed:e.seed,difficulty:e.difficulty,budget:e.budget,source:e.provenance,final_evaluation:e.result},null,2));
    const isModel=model() && model().kind==="model";
    text("policy-note",e.provenance.privileged_policy?"Oracle reference · privileged state":isModel?"Actual "+model().name+" · recorded model decisions":"Manual / scripted interaction");
    text("model-output",f.model_output || "No model call at the initial observation.");
    const input=f.model_input;
    text("model-input-label",input?"Model input":state.step?"Public input (historical)":"Task instruction");
    text("model-input",input?(input.text?input.text+"\n\nInput views: "+input.image_streams.join(", ")+" · before step "+input.image_observation_step:JSON.stringify(input,null,2)):state.step?"Exact model request was not retained in this historical record. Public observation before this action:\n"+JSON.stringify(before.observation,null,2):f.observation.instruction||e.instruction);
    $("model-output-details").hidden=!f.model_output;
    $("model-exchange").hidden=!isModel&&!input&&!f.model_output;
    $("model-exchange").classList.toggle("input-only",!f.model_output);
    const finished=state.step===e.frames.length-1&&(!state.processSample||obs.phase==="complete");
    const unfinished=e.agent==="online-session"&&!e.result.terminated&&!e.result.truncated;
    const running=unfinished&&(!e.worker_status||e.worker_status==="RUNNING");
    text("episode-status",unfinished&&!running?"Worker "+e.worker_status.toLowerCase()+" · task not evaluated":running?"Online session · running":finished?(e.result.success===null?"Outcome not evaluated / recording cutoff":e.result.success?"Verified success":e.result.truncated?"Budget exhausted · not successful":"Submitted · strict success not met"):"Recorded episode · step "+state.step);
    $("episode-status").className="episode-status"+(finished&&!running?(e.result.success?" success":" failure"):"");
  }

  function seek(value, stop) {
    const e=episode();
    if (!e) return false;
    if (stop!==false) pause();
    const index=Math.max(0,Math.min(e.frames.length-1,Math.round(Number(value)||0)));
    if(e.process&&state.processMode==="continuous"){
      const samples=e.process.samples;
      const sample=index===e.frames.length-1?samples[samples.length-1]:samples.find(s=>s.action_index>=index)||samples[samples.length-1];
      $("process-fpv").currentTime=sample.time_s-e.process.time_offset_s;
      syncProcess();return true;
    }
    state.processSample=null;state.processMode="decisions";
    state.step=index;
    renderFrame();
    return true;
  }

  function loadCatalog(data) {
    validate(data); pause(); catalog=data;
    onlineEpisode=null;
    $("environment-select").innerHTML='<option value="all">All environments</option>'+catalog.environments.map(e=>'<option value="'+escape(e.id)+'">'+escape(e.name)+'</option>').join("");
    state.environment="all"; state.agent=catalog.default_agent || catalog.models[0].id;
    chooseTask(catalog.tasks.some(t=>t.id===catalog.default_task)?catalog.default_task:catalog.tasks[0].id);
    const modelEpisodes=catalog.episodes.filter(e=>catalog.models.some(m=>m.id===e.agent&&m.kind==="model"));
    text("stat-tasks",catalog.tasks.filter(t=>t.episode_ids.length).length);
    text("stat-episodes",catalog.episodes.length);
    text("stat-model",modelEpisodes.length);
  }

  function appendOnline(snapshot, reset, response) {
    pause();
    const taskId="online_"+snapshot.environment+"_"+snapshot.task;
    const environment=snapshot.environment==="native"?"ai2thor":snapshot.environment==="deliverybench"?"simworld_delivery":snapshot.environment;
    if(!catalog.models.some(m=>m.id==="online-session")) catalog.models.push({id:"online-session",name:"Online session (per-step policy)",kind:"session",checkpoint:null});
    if(!catalog.tasks.some(t=>t.id===taskId)) catalog.tasks.push({id:taskId,title:title(snapshot.task)+" (online)",category:"Online environment",summary:snapshot.observation.instruction||"Online interaction",objectives:[snapshot.observation.instruction||"Follow the environment objective"],environment,research:(catalog.tasks.find(t=>t.id===snapshot.task)||{}).research,episode_ids:[]});
    if(reset || !onlineEpisode || onlineEpisode.task!==taskId){
      onlineEpisode={id:taskId+"-"+Date.now(),task:taskId,agent:"online-session",scene:snapshot.scene||snapshot.environment,seed:snapshot.seed,difficulty:snapshot.difficulty||0,budget:snapshot.budget||100,instruction:snapshot.observation.instruction||"",frames:[],result:{success:null,metrics:{},terminated:false,truncated:false},provenance:{directory:"Online session log",policy:"Per-step policy identity",privileged_policy:false}};
      catalog.episodes.push(onlineEpisode); catalog.tasks.find(t=>t.id===taskId).episode_ids.push(onlineEpisode.id);
    }
    const previous=onlineEpisode.frames[onlineEpisode.frames.length-1],replace=previous&&previous.environment_step===snapshot.step;
    const f={step:replace?previous.step:onlineEpisode.frames.length,environment_step:snapshot.step,total_reward:snapshot.total_reward,image:(snapshot.views.find(v=>v.name==="First-person RGB")||snapshot.views[0]).image,views:snapshot.views,action:snapshot.action||null,reward:snapshot.reward||0,terminated:snapshot.terminated,truncated:snapshot.truncated,observation:snapshot.observation,agent:snapshot.policy||"Manual / state read",model_output:response||snapshot.model_output||(replace?previous.model_output:""),model_input:snapshot.model_input||(replace?previous.model_input:null)};
    if(replace)onlineEpisode.frames[f.step]=f;else onlineEpisode.frames.push(f);
    onlineEpisode.result={success:snapshot.evaluation&&typeof snapshot.evaluation.success==="boolean"?snapshot.evaluation.success:null,metrics:snapshot.evaluation||{},terminated:snapshot.terminated,truncated:snapshot.truncated};
    onlineEpisode.worker_status=snapshot.run_status||null;
    state.environment="all"; state.agent="online-session"; chooseTask(taskId); chooseEpisode(onlineEpisode.id); seek(f.step);
  }

  $("process-mode").addEventListener("change",e=>{state.processMode=e.target.value;state.processSample=null;renderFrame();if(state.processMode==="continuous")syncProcess();});
  $("process-fpv").addEventListener("timeupdate",syncProcess);
  $("process-fpv").addEventListener("seeked",syncProcess);
  $("process-fpv").addEventListener("play",()=>["process-global","process-local"].forEach(id=>{$(id).currentTime=$("process-fpv").currentTime;const promise=$(id).play();if(promise)promise.catch(()=>{});}));
  $("process-fpv").addEventListener("pause",()=>["process-global","process-local"].forEach(id=>$(id).pause()));
  $("process-speed").addEventListener("change",e=>["process-fpv","process-global","process-local"].forEach(id=>$(id).playbackRate=Number(e.target.value)));
  $("environment-select").addEventListener("change",e=>chooseEnvironment(e.target.value));
  $("agent-select").addEventListener("change",e=>chooseAgent(e.target.value));
  $("episode-select").addEventListener("change",e=>chooseEpisode(e.target.value));
  $("view-select").addEventListener("change",e=>{state.view=e.target.value;renderFrame();});
  $("task-select").addEventListener("change",e=>chooseTask(e.target.value));
  $("timeline").addEventListener("input",e=>seek(e.target.value));
  $("previous-button").addEventListener("click",()=>seek(state.step-1));
  $("next-button").addEventListener("click",()=>seek(state.step+1));
  $("play-button").addEventListener("click",play);
  $("end-button").addEventListener("click",()=>{if(episode()) seek(episode().frames.length-1);});
  $("speed-select").addEventListener("change",()=>{if(state.timer!==null){pause();play();}});
  document.addEventListener("visibilitychange",()=>{if(document.hidden) pause();});
  document.addEventListener("keydown",e=>{
    if (e.target && /INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY/.test(e.target.tagName)) return;
    if (e.key==="ArrowRight"){e.preventDefault();seek(state.step+1);}
    if (e.key==="ArrowLeft"){e.preventDefault();seek(state.step-1);}
    if (e.code==="Space"){e.preventDefault();play();}
    if (e.key==="Escape") pause();
  });
  window.SpatialDemo={loadCatalog,chooseTask,chooseEnvironment,chooseAgent,chooseEpisode,seek,pause,play,appendOnline,
    getState:()=>({task:state.task,agent:state.agent,episode:state.episode,step:state.step,environment:state.environment,playing:state.timer!==null}),
    describeAction};
  try { loadCatalog(window.SPATIAL_CATALOG); }
  catch(error){text("task-title","Episode catalog could not be loaded");text("selection-notice",error.message);throw error;}
}());
