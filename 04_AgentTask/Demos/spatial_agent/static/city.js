(function () {
  "use strict";
  const $=id=>document.getElementById(id),data=window.SPATIAL_CITIES;
  const svg=$("city-map"),scene=$("city-scene");
  let position={x:0,y:0},target=null,path=[],distance=0,zoom=1,pan={x:0,y:0},drag=null,moved=false,recordedTrace=[];
  const map=()=>data.maps[$("city-select").value];
  function shortest(roads,start,end){
    const nodes=[],edges=[];
    function index(p){let i=nodes.findIndex(n=>Math.hypot(n.x-p.x,n.y-p.y)<.001);if(i<0){i=nodes.length;nodes.push(p);edges.push([]);}return i;}
    roads.forEach(r=>{const a=index(r.start),b=index(r.end),d=Math.hypot(r.start.x-r.end.x,r.start.y-r.end.y);edges[a].push([b,d]);edges[b].push([a,d]);});
    if(!nodes.length)throw new Error("Map has no roads");
    const nearest=p=>nodes.reduce((best,n,i)=>Math.hypot(n.x-p.x,n.y-p.y)<Math.hypot(nodes[best].x-p.x,nodes[best].y-p.y)?i:best,0);
    const source=nearest(start),goal=nearest(end),cost=nodes.map(()=>Infinity),previous=[],seen=new Set();cost[source]=0;
    while(seen.size<nodes.length){let u=-1;nodes.forEach((_,i)=>{if(!seen.has(i)&&(u<0||cost[i]<cost[u]))u=i;});if(u<0||!Number.isFinite(cost[u]))break;seen.add(u);edges[u].forEach(([v,w])=>{if(cost[u]+w<cost[v]){cost[v]=cost[u]+w;previous[v]=u;}});}
    if(!Number.isFinite(cost[goal]))throw new Error("Road components are disconnected");
    const ids=[goal];while(ids[0]!==source)ids.unshift(previous[ids[0]]);
    const points=[start,...ids.map(i=>nodes[i]),end];
    return {points,distance:points.slice(1).reduce((sum,p,i)=>sum+Math.hypot(p.x-points[i].x,p.y-points[i].y),0)};
  }
  function color(type){const name=String(type).toLowerCase();return name.includes("restaurant")?"#d67828":name.includes("charg")?"#23936c":name.includes("store")?"#8855bb":"#387caf";}
  function select(p){target=p;path=[];distance=0;$("city-selection").textContent=(p.type||"Selected point")+" · "+(p.id||"")+" · "+p.x.toFixed(1)+", "+p.y.toFixed(1)+" m";draw();}
  function draw(){
    const points=map().roads.reduce((items,r)=>items.concat([r.start,r.end]),[]),xs=points.map(p=>p.x),ys=points.map(p=>p.y);
    const minx=Math.min(...xs)-100,miny=Math.min(...ys)-100,width=Math.max(...xs)-minx+100,height=Math.max(...ys)-miny+100;
    const scale=Math.min(900/width,560/height);
    scene.innerHTML="";scene.setAttribute("transform",`translate(${50+pan.x},${35+pan.y}) scale(${zoom*scale}) translate(${-minx},${-miny})`);
    function add(tag,attrs,parent=scene){const node=document.createElementNS("http://www.w3.org/2000/svg",tag);Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,value));parent.append(node);return node;}
    if($("city-buildings").checked)map().buildings.forEach(b=>{const r=b.bounds;add("rect",{x:r.x,y:r.y,width:r.width,height:r.height,fill:"#e8e8e8",stroke:"#aaa",transform:`rotate(${r.rotation||0} ${b.center.x} ${b.center.y})`});});
    map().roads.forEach(r=>add("line",{x1:r.start.x,y1:r.start.y,x2:r.end.x,y2:r.end.y,stroke:"#658dab","stroke-width":8}));
    if(recordedTrace.length)add("polyline",{points:recordedTrace.map(p=>p.x+","+p.y).join(" "),fill:"none",stroke:"#8a4c91","stroke-width":4});
    if(path.length)add("polyline",{points:path.map(p=>p.x+","+p.y).join(" "),fill:"none",stroke:"#c87420","stroke-width":5});
    if($("city-pois").checked)map().pois.forEach(p=>{const circle=add("circle",{cx:p.x,cy:p.y,r:5,fill:color(p.type)});add("title",{},circle).textContent=p.type+" · "+p.id;circle.onclick=e=>{e.stopPropagation();select(p);};});
    add("circle",{cx:position.x,cy:position.y,r:8,fill:"#258755"});
    if(target)add("circle",{cx:target.x,cy:target.y,r:10,fill:"none",stroke:"#bb532a","stroke-width":2});
    const m=data.mechanics,battery=Math.max(0,Math.min(100,Number($("city-battery").value)));
    $("city-state").textContent=JSON.stringify({map:$("city-select").value,roads:map().roads.length,buildings:map().buildings.length,pois:map().pois.length,diagram_position_m:position,road_centre_distance_m:distance,travel_estimates:["walk","e-scooter","car"].map(mode=>({mode,minutes:distance/(m.avg_speed_cm_s[mode]/100)/60,energy_percent:distance*m.energy_pct_decay_per_m_by_mode[mode]})),scooter_battery_required_percent:distance*m.scooter_batt_decay_pct_per_m,illustrative_initial_battery_percent:battery,sufficient:distance*m.scooter_batt_decay_pct_per_m<=battery,scope:"Nearest-junction approximation; excludes native collisions, traffic, waits, order events, and settlement. Diagram changes do not change the simulator."},null,2);
  }
  function reset(){recordedTrace=[];position={x:0,y:0};target=null;path=[];distance=0;zoom=1;pan={x:0,y:0};$("city-selection").textContent="Select a POI or map point. Drag to pan; scroll to zoom.";draw();}
  $("city-select").innerHTML=Object.keys(data.maps).map(name=>'<option>'+name+'</option>').join("");
  $("city-select").value="small-city-11";$("city-api").textContent=data.api;
  ["city-select","city-reset"].forEach(id=>$(id).addEventListener(id==="city-select"?"change":"click",reset));
  ["city-buildings","city-pois","city-battery"].forEach(id=>$(id).addEventListener("change",draw));
  $("city-route").addEventListener("click",()=>{try{if(!target)throw new Error("Select a target first");const route=shortest(map().roads,position,target);path=route.points;distance=route.distance;draw();}catch(error){$("city-selection").textContent=error.message;}});
  $("city-move").addEventListener("click",()=>{if(target){position={x:target.x,y:target.y};path=[];distance=0;draw();}});
  $("city-send").addEventListener("click",()=>{if(target){$("online-action").value=`MOVE(${target.x.toFixed(1)}m, ${target.y.toFixed(1)}m)`;$("online-controls").open=true;$("online-controls").scrollIntoView();}});
  svg.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};moved=false;};
  svg.onpointermove=e=>{if(!drag)return;const dx=(e.clientX-drag.x)*1000/svg.clientWidth,dy=(e.clientY-drag.y)*650/svg.clientHeight;if(Math.abs(dx)+Math.abs(dy)>5)moved=true;if(moved){pan={x:drag.px+dx,y:drag.py+dy};draw();}};
  svg.onpointerup=()=>{drag=null;};svg.onpointerleave=()=>{drag=null;};
  svg.onclick=e=>{if(moved){moved=false;return;}const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(scene.getScreenCTM().inverse());select({x:p.x,y:p.y});};
  svg.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(6,zoom*(e.deltaY>0?.9:1.1)));draw();};
  function setRecordedState(name,state,trail){
    if(!data.maps[name])return;
    if($("city-select").value!==name){$("city-select").value=name;reset();}
    $("city-recorded-state").textContent=JSON.stringify(Object.assign({source:"Recorded native state; diagram controls do not change the simulator",map:name},state),null,2);
    recordedTrace=(trail||[]).map(p=>({x:p[0]/100,y:p[1]/100}));
    if($("city-follow").checked&&state.position_cm){
      const p=state.position_cm;position={x:(Array.isArray(p)?p[0]:p.x)/100,y:(Array.isArray(p)?p[1]:p.y)/100};
    }
    draw();
  }
  window.CityDemo={shortest,setRecordedState};
  reset();
}());
