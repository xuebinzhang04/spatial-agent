"""Evidence and UI-state checks. QJSEngine runs real JS against a small DOM stub.

This does not claim browser layout or screenshot validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtQml import QJSEngine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "site"
RECORD = ROOT / "05_Experiment/results" / ("spatial-agent-demo-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
APP = QCoreApplication.instance() or QCoreApplication([])


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = {}
        self.duplicates = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.items:
                self.duplicates.append(attrs["id"])
            self.items[attrs["id"]] = {"tagName": tag.upper(), "className": attrs.get("class", ""), "value": attrs.get("value", "")}
        if tag in ("script", "link", "source") or tag == "img" and "src" in attrs:
            self.links.append(attrs.get("src", attrs.get("href", "")))


DOM = r"""
var window=this, __elements={}, __timers={}, __timerIndex=0;
function Element(info){
 this.tagName=info.tagName;this.className=info.className;this.value=info.value;
 this.textContent='';this.innerHTML='';this.hidden=false;this.disabled=false;
 this.attributes={};this.listeners={};this.dataset={};this.offsetWidth=500;
 var names=new Set(this.className.split(' ').filter(Boolean));
 this.classList={add:function(x){names.add(x);},remove:function(x){names.delete(x);},contains:function(x){return names.has(x);},toggle:function(x,force){var on=force===undefined?!names.has(x):force;if(on)names.add(x);else names.delete(x);return on;}};
}
Element.prototype.setAttribute=function(k,v){this.attributes[k]=v;};
Element.prototype.addEventListener=function(k,f){this.listeners[k]=f;};
Element.prototype.scrollIntoView=function(){};
Element.prototype.showModal=function(){this.open=true;};
Element.prototype.close=function(){this.open=false;};
Element.prototype.pause=function(){};
Element.prototype.load=function(){};
Element.prototype.removeAttribute=function(k){delete this.attributes[k];if(k==='src')this.src='';};
Element.prototype.click=function(){if(!this.disabled&&this.listeners.click)this.listeners.click({target:this});};
var document={body:new Element({tagName:'BODY',className:'',value:''}),hidden:false,getElementById:function(id){if(!__elements[id])throw new Error('Missing DOM id: '+id);return __elements[id];},listeners:{},addEventListener:function(k,f){this.listeners[k]=f;}};
function setInterval(f){__timers[++__timerIndex]=f;return __timerIndex;}
function clearInterval(id){delete __timers[id];}
"""


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((OUTPUT / "data/catalog.json").read_text())
        cls.dom = Elements()
        cls.dom.feed((OUTPUT / "index.html").read_text())

    def setUp(self):
        self.engine = QJSEngine()
        self.evaluate(DOM)
        for key, info in self.dom.items.items():
            self.evaluate("__elements[" + json.dumps(key) + "]=new Element(" + json.dumps(info) + ");")
        self.evaluate("__elements['speed-select'].value='750';")
        self.evaluate((OUTPUT / "catalog.js").read_text())
        self.evaluate((OUTPUT / "app.js").read_text())

    def evaluate(self, script):
        result = self.engine.evaluate(script)
        if result.isError():
            self.fail(result.toString() + " at " + result.property("lineNumber").toString() + " " + result.property("stack").toString())
        return result

    def js(self, expression):
        return json.loads(self.evaluate("JSON.stringify(" + expression + ")").toString())

    def test_generation_audit_renders_every_task_variable_model(self):
        self.evaluate((OUTPUT / "generation.js").read_text())
        self.evaluate((OUTPUT / "static/generation.js").read_text())
        model = json.loads((ROOT / "04_AgentTask/CustomAgentTasks/task_variables.json").read_text())
        self.assertEqual(self.js("__elements['generation-task'].value"), next(iter(model["tasks"])))
        audited = {case["task"] for audit in self.js("SPATIAL_GENERATION.audits") for case in audit["cases"]}
        for task, spec in model["tasks"].items():
            self.evaluate("__elements['generation-task'].value=" + json.dumps(task) + ";__elements['generation-task'].listeners.change();")
            html = self.js("__elements['generation-body'].innerHTML")
            for stage in ("build", "update", "freeze"):
                self.assertIn(stage, html)
            for name in spec["global_memory"]:
                self.assertIn(name.replace("_", " "), html)
            if task in audited:
                self.assertIn("<td>" + task, html)  # at least one generated case row

    def test_case_link_opens_its_episode_in_the_explorer(self):
        self.evaluate((OUTPUT / "static/cases.js").read_text())
        examples=json.loads((OUTPUT/'data/gif-examples.json').read_text())
        item=next(e for e in examples if e['task']=='memory_maze')
        self.evaluate("var link={dataset:{task:'memory_maze',episode:"+json.dumps(item['episode'])+"}};link.closest=function(){return link;};"
                      "document.listeners.click({target:link});")
        self.assertEqual(self.js("SpatialDemo.getState().episode"),item['episode'])

    def test_initial_model_is_actual_recorded_model(self):
        self.assertEqual(self.js("SpatialDemo.getState().agent"), "Qwen3.8-27B")
        self.assertEqual(self.js("document.getElementById('state-agent').textContent"), "Qwen3.8-27B")
        self.assertIn("Actual Qwen", self.js("document.getElementById('policy-note').textContent"))

    def test_prompt_output_precede_media_and_difficulty_stays_beside_it(self):
        html=(OUTPUT/'index.html').read_text()
        self.assertLess(html.index('id="model-input"'),html.index('id="viewport"'))
        self.assertLess(html.index('id="model-output"'),html.index('id="viewport"'))
        sidebar=html.split('<aside class="interaction-state"',1)[1].split('</aside>',1)[0]
        self.assertNotIn('id="model-input"',sidebar)
        self.assertIn('id="state-resources"',sidebar)
        self.assertIn('id="case-difficulty"',sidebar)

    def test_online_response_and_delivery_map_have_their_own_entries(self):
        html=(OUTPUT/'index.html').read_text()
        explorer=html.split('<section id="explorer"',1)[1].split('<section id="online-demo"',1)[0]
        self.assertNotIn('id="online-model-response"',explorer)
        self.assertIn('id="city-map"',explorer)
        self.evaluate("SpatialDemo.chooseEnvironment('simworld_delivery')")
        self.assertFalse(self.js("document.getElementById('delivery-map-tools').hidden"))
        self.evaluate("SpatialDemo.chooseTask('shape_identification')")
        self.assertTrue(self.js("document.getElementById('delivery-map-tools').hidden"))

    def test_full_delivery_process_tracks_time_without_inventing_decisions(self):
        episodes=[e for e in self.catalog['episodes'] if e.get('process')]
        if not episodes:self.skipTest('Continuous model process has not been imported')
        e=episodes[-1];samples=e['process']['samples']
        self.assertGreater(len(samples),10*(len(e['frames'])-1))
        self.assertTrue(all(b['time_s']>=a['time_s'] for a,b in zip(samples,samples[1:])))
        moving=[s for s in samples if s['action_index']==2]
        self.assertGreater(len({tuple(s['state']['position_cm']) for s in moving}),10)
        self.evaluate("SpatialDemo.chooseTask("+json.dumps(e['task'])+");SpatialDemo.chooseAgent('Qwen3.8-27B');SpatialDemo.chooseEpisode("+json.dumps(e['id'])+")")
        self.assertFalse(self.js("document.getElementById('delivery-process').hidden"))
        self.assertTrue(self.js("document.getElementById('viewport').hidden"))
        sample=moving[len(moving)//2]
        self.evaluate("__elements['process-fpv'].currentTime="+str(sample['time_s']-e['process']['time_offset_s'])+";__elements['process-fpv'].listeners.timeupdate()")
        self.assertEqual(self.js("SpatialDemo.getState().step"),2)
        self.assertIn('Battery',self.js("document.getElementById('state-resources').textContent"))
        self.assertNotIn('Verified success',self.js("document.getElementById('episode-status').textContent"))
        self.evaluate("SpatialDemo.seek(3)")
        self.assertFalse(self.js("document.getElementById('delivery-process').hidden"))
        self.assertGreater(self.js("document.getElementById('process-fpv').currentTime"),0)
        self.evaluate("__elements['process-mode'].listeners.change({target:{value:'decisions'}})")
        self.assertTrue(self.js("document.getElementById('delivery-process').hidden"))
        self.assertFalse(self.js("document.getElementById('viewport').hidden"))

    def test_fixed_delivery_replay_keeps_model_inputs_bound_to_the_action(self):
        episodes=[e for e in self.catalog['episodes'] if e['task']=='delivery_pickup_delivery' and e['agent']=='Qwen3.8-27B']
        if not episodes:self.skipTest('A completed fixed-case Qwen run has not been imported yet')
        e=episodes[-1]
        self.evaluate("SpatialDemo.chooseTask('delivery_pickup_delivery');SpatialDemo.chooseAgent('Qwen3.8-27B');SpatialDemo.chooseEpisode("+json.dumps(e['id'])+");SpatialDemo.seek(1)")
        frame=e['frames'][1]
        self.assertEqual(json.loads(frame['model_output'])['command'],frame['action'])
        text=self.js("document.getElementById('model-input').textContent")
        self.assertIn(frame['model_input']['text'],text)
        self.assertIn('First-person RGB',text)
        self.assertNotIn('serial_witness',text)
        self.assertNotIn('center_cm',text)
        self.assertIn('deadline_s',self.js("document.getElementById('case-difficulty').innerHTML"))
        calls=[json.loads(line) for line in (ROOT/e['provenance']['directory']/'model-calls.jsonl').read_text().splitlines()]
        import base64
        for frame,call in zip(e['frames'][1:],calls):
            self.assertEqual(frame['model_input']['text'],call['request']['messages'][0]['content'][0]['text'])
            raw_images=[part['image_url']['url'] for part in call['request']['messages'][0]['content'] if part['type']=='image_url']
            for view,raw in zip(frame['model_input']['views'],raw_images):
                self.assertEqual((OUTPUT/view['image']).read_bytes(),base64.b64decode(raw.split(',',1)[1]))

    def test_task_switch_falls_back_to_correct_policy(self):
        self.evaluate("SpatialDemo.chooseTask('dynamic_replanning')")
        self.assertEqual(self.js("SpatialDemo.getState().agent"), "oracle")
        self.assertIn("Oracle reference", self.js("document.getElementById('state-agent').textContent"))
        self.assertFalse(self.js("SpatialDemo.chooseAgent('Qwen3.8-27B')"))
        self.assertIn("no imported rollout", self.js("document.getElementById('selection-notice').textContent"))

    def test_environment_and_case_switch(self):
        self.evaluate("__elements['environment-select'].value='shape3d';__elements['environment-select'].listeners.change({target:__elements['environment-select']});")
        self.assertEqual(self.js("SpatialDemo.getState().task"), "shape_identification")
        self.evaluate("SpatialDemo.chooseTask('dependency_doors')")
        ids = self.js("SPATIAL_CATALOG.episodes.filter(function(e){return e.task==='dependency_doors';}).map(function(e){return e.id;})")
        self.assertGreaterEqual(len(ids), 2)  # historical L1-L3 plus the current-generator showcase case
        self.evaluate("SpatialDemo.chooseEpisode(" + json.dumps(ids[-1]) + ")")
        self.assertEqual(self.js("SpatialDemo.getState().episode"), ids[-1])

    def test_step_corresponds_to_the_single_recorded_view(self):
        self.evaluate("document.getElementById('next-button').click()")
        self.assertEqual(self.js("SpatialDemo.getState().step"), 1)
        e = next(e for e in self.catalog["episodes"] if e["id"] == self.js("SpatialDemo.getState().episode"))
        self.assertEqual(self.js("document.getElementById('observation-image').src"), e["frames"][1]["image"])
        self.assertEqual(self.js("document.getElementById('action-label').textContent"), "Commit memory")
        self.assertTrue(self.js("document.getElementById('observation-phase').textContent").startswith('Discern: '))
        self.assertEqual(self.js("document.getElementById('observation-phase').textContent"),e['presentation_phase_notes']['detection'])

    def test_high_reward_is_not_reported_as_success(self):
        self.evaluate("document.getElementById('end-button').click()")
        self.assertIn("strict success not met", self.js("document.getElementById('episode-status').textContent"))
        self.assertNotIn("complete", self.js("document.getElementById('objective-list').innerHTML"))

    def test_transition_details_preserve_submitted_answer(self):
        self.evaluate("document.getElementById('end-button').click()")
        episode = next(e for e in self.catalog["episodes"] if e["id"] == self.js("SpatialDemo.getState().episode"))
        record = json.loads(self.js("document.getElementById('step-record').textContent"))
        self.assertEqual(record["action"], episode["frames"][-1]["action"])
        self.assertIn("answer", record["action"])
        self.assertTrue(record["terminated"])
        self.assertEqual(record["observation"], episode["frames"][-1]["observation"])
        self.assertFalse(self.js("SpatialDemo.chooseTask('signal_repair')"))

    def test_budget_exhaustion_and_oracle_success(self):
        self.evaluate("SpatialDemo.chooseTask('shape_identification');document.getElementById('end-button').click()")
        self.assertIn("Budget exhausted", self.js("document.getElementById('episode-status').textContent"))
        self.evaluate("SpatialDemo.chooseAgent('oracle');document.getElementById('end-button').click()")
        self.assertEqual(self.js("document.getElementById('episode-status').textContent"), "Verified success")
        self.assertTrue(self.js("document.getElementById('model-output-details').hidden"))

    def test_unrecorded_task_has_no_fabricated_actions(self):
        self.evaluate("var pending=JSON.parse(JSON.stringify(SPATIAL_CATALOG));pending.tasks.push({id:'pending_task',title:'Pending task',category:'Pending',summary:'No recording',objectives:['Goal'],environment:'shape3d',episode_ids:[]});SpatialDemo.loadCatalog(pending);SpatialDemo.chooseTask('pending_task')")
        self.assertIsNone(self.js("SpatialDemo.getState().episode"))
        self.assertTrue(self.js("document.getElementById('play-button').disabled"))
        self.assertFalse(self.js("document.getElementById('empty-state').hidden"))
        self.assertFalse(self.js("SpatialDemo.seek(1)"))

    def test_play_pause_and_final_step(self):
        self.evaluate("SpatialDemo.play();__timers[__timerIndex]()")
        self.assertEqual(self.js("SpatialDemo.getState().step"), 1)
        self.assertTrue(self.js("SpatialDemo.getState().playing"))
        self.evaluate("SpatialDemo.pause();SpatialDemo.seek(100000)")
        self.assertFalse(self.js("SpatialDemo.getState().playing"))
        self.assertTrue(self.js("document.getElementById('next-button').disabled"))

    def test_recording_details_replace_extra_controls(self):
        self.assertIn("trajectory_sha256", self.js("document.getElementById('provenance-content').textContent"))
        for removed in ('focus-button','view-current','view-compare','previous-wrap','all-views',
                        'model-input-views','export-record','provenance-dialog','mechanism-type'):
            self.assertNotIn(removed,self.dom.items)
        html=(OUTPUT/'index.html').read_text()
        self.assertNotIn('Environment Variables and Mechanism Examples',html)
        self.assertNotIn('static/mechanisms.js',html)
        self.assertNotIn('mechanisms',self.catalog)
        self.assertTrue(self.js("document.getElementById('decision-view-controls').hidden"))
        self.evaluate("SpatialDemo.chooseTask('dependency_doors')")
        self.assertTrue(self.js("document.getElementById('model-exchange').hidden"))

    def test_catalog_extension_uses_same_interface(self):
        self.evaluate("var extra=JSON.parse(JSON.stringify(SPATIAL_CATALOG));extra.tasks.push({id:'new_task',title:'New task',category:'New family',summary:'Added via the adapter',objectives:['Goal'],environment:'shape3d',episode_ids:[]});SpatialDemo.loadCatalog(extra);SpatialDemo.chooseTask('new_task')")
        self.assertEqual(self.js("document.getElementById('task-title').textContent"), "New task")
        self.assertFalse(self.js("document.getElementById('empty-state').hidden"))

    def test_source_images_and_public_data(self):
        hashes = {}
        assets = json.loads((OUTPUT / "data/evidence.json").read_text())["web_assets"]
        for e in self.catalog["episodes"]:
            self.assertEqual(e["agent"] == "oracle", e["provenance"]["privileged_policy"])
            if e["provenance"].get("recording_complete") is not False:
                self.assertTrue(e["frames"][-1]["terminated"] or e["frames"][-1]["truncated"])
            for f in e["frames"]:
                self.assertFalse(set(f["observation"]) & {"ground_truth", "privileged_state", "witness", "verification"})
                if "source_rgb_sha256" in f:
                    self.assertEqual(set(f["observation"]), {"instruction", "phase", "memory", "feedback", "view_step", "memory_checkpoint", "task_progress"})
                    if f["image"] not in hashes:
                        with Image.open(OUTPUT / f["image"]) as image:
                            hashes[f["image"]] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                    self.assertEqual(hashes[f["image"]], f.get('display_rgb_sha256',f["source_rgb_sha256"]))
                    if f.get('recorded_image'):
                        with Image.open(OUTPUT/f['recorded_image']) as original:
                            self.assertEqual(hashlib.sha256(original.convert('RGB').tobytes()).hexdigest(),f['source_rgb_sha256'])
                else:
                    for view in f.get("views", [{"image": f["image"]}]):
                        source = OUTPUT / view["image"]
                        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), assets[view["image"]])
        self.assertFalse(self.dom.duplicates)
        for path in self.dom.links:
            self.assertTrue((OUTPUT / path).exists(), path)

    def test_deliverybench_streams_and_cutoff_are_preserved(self):
        self.evaluate("SpatialDemo.chooseTask('deliverybench')")
        self.assertEqual(self.js("SpatialDemo.getState().agent"), "manual-script")
        self.assertIn("Global map", self.js("document.getElementById('view-select').innerHTML"))
        self.assertIn("battery_pct", self.js("document.getElementById('step-record').textContent"))
        self.evaluate("document.getElementById('end-button').click()")
        self.assertIn("not evaluated", self.js("document.getElementById('episode-status').textContent"))

    def test_graph_prototype_removed_and_delivery_defaults_to_full_process(self):
        self.assertNotIn('symbolic', {e['id'] for e in self.catalog['environments']})
        self.assertFalse(any(t['id'].startswith('symbolic_') for t in self.catalog['tasks']))
        self.evaluate("SpatialDemo.chooseTask('dependency_doors');SpatialDemo.chooseEnvironment('simworld_delivery')")
        self.assertEqual(self.js("SpatialDemo.getState().episode"), 'delivery-qwen-389912')
        self.assertEqual(self.js("SpatialDemo.getState().agent"), 'Qwen3.8-27B')
        self.assertFalse(self.js("document.getElementById('delivery-process').hidden"))
        self.assertTrue(self.js("document.getElementById('decision-playback').hidden"))
        html = (OUTPUT/'index.html').read_text()
        self.assertLess(html.index('id="online-infer"'), html.index('id="online-model-response"'))

    def test_online_frames_preserve_per_step_policy(self):
        self.evaluate("SpatialDemo.appendOnline({environment:'shape3d',task:'shape_identification',step:0,seed:7,budget:20,views:[{name:'Agent observation',image:'test.png'}],observation:{instruction:'Test goal'},reward:0,terminated:false,truncated:false},true)")
        self.evaluate("SpatialDemo.appendOnline({environment:'shape3d',task:'shape_identification',step:1,seed:7,budget:20,policy:'Qwen3.8-27B',action:{action:'CAMERA_RIGHT'},views:[{name:'Agent observation',image:'test2.png'}],observation:{instruction:'Test goal'},reward:0,terminated:false,truncated:false},false,'model response')")
        self.assertEqual(self.js("document.getElementById('state-agent').textContent"), "Qwen3.8-27B")
        self.assertEqual(self.js("document.getElementById('observation-image').src"), "test2.png")
        self.evaluate("SpatialDemo.seek(0)")
        self.assertIn("Manual", self.js("document.getElementById('state-agent').textContent"))

    def test_merged_demo_definitions_follow_player_selection(self):
        self.assertIn("<code>MEMORIZE</code>", self.js("document.getElementById('demo-actions').innerHTML"))
        self.assertIn("<code>previous_location</code>", self.js("document.getElementById('demo-verification').innerHTML"))
        self.assertIn("<code>scene_id</code>", self.js("document.getElementById('demo-variables').innerHTML"))
        self.evaluate("SpatialDemo.chooseEnvironment('simworld_delivery')")
        self.assertEqual(self.js("SpatialDemo.getState().task"), "delivery_pickup_delivery")
        self.assertIn("5.3.2", self.js("document.getElementById('demo-simulator').textContent"))
        self.assertIn("Accept each order", self.js("document.getElementById('demo-objective').textContent"))
        self.evaluate("SpatialDemo.chooseEnvironment('shape3d');SpatialDemo.chooseTask('pose_alignment');SpatialDemo.chooseAgent('Qwen3.8-27B')")
        episode_id = self.js("SpatialDemo.getState().episode")
        self.evaluate("SpatialDemo.seek(999)")
        self.assertEqual(self.js("SpatialDemo.getState().episode"), episode_id)
        self.assertEqual(self.js("document.getElementById('demo-verification').textContent"), self.js("SPATIAL_CATALOG.tasks.find(t=>t.id==='pose_alignment').research.verification"))
        self.evaluate("SpatialDemo.chooseAgent('oracle')")
        self.assertNotEqual(self.js("SpatialDemo.getState().episode"), episode_id)

    def test_example_gallery_restores_phase_labels_without_object_numbers(self):
        examples=json.loads((OUTPUT/'data/gif-examples.json').read_text())
        self.assertEqual(len(examples),8)
        self.assertEqual(len({item['task'] for item in examples}),8)
        for item in examples:
            episode=next(e for e in self.catalog['episodes'] if e['id']==item['episode'])
            self.assertTrue(episode['provenance']['privileged_policy'])
            if not item['displayed']:
                self.assertEqual(item['status'],'pending_clean_recording')
                self.assertIsNone(item['asset'])
                continue
            self.assertFalse(item['object_number_overlays'])
            self.assertTrue(item['added_annotations'])
            self.assertEqual(item['annotation_kind'],'phase_only_top_right')
            self.assertGreaterEqual(max(item['source_resolution_px']),512)
            self.assertEqual(item['steps'],sorted(set(item['steps'])))
            with Image.open(OUTPUT/item['asset']) as gif:
                self.assertGreater(gif.n_frames,1)
                self.assertEqual(gif.size,(640,640))
                self.assertEqual(gif.format,'GIF')
                for i in range(gif.n_frames):
                    gif.seek(i)
                    self.assertGreaterEqual(gif.info['duration'],1600)
        html=(OUTPUT/'index.html').read_text()
        cases=html.split('<section id="tasks"',1)[1].split('</section>',1)[0]
        self.assertEqual(cases.count('class="case-card"'),8)
        for item in examples:
            self.assertIn('src="'+item['asset']+'"',cases)
            self.assertIn('data-episode="'+item['episode']+'"',cases)
        self.assertEqual(cases.count('<dl class="case-facts">'),8)
        for stale in (OUTPUT/'static/media').glob('*.gif'):
            self.assertIn(str(stale.relative_to(OUTPUT)),{e['asset'] for e in examples if e['displayed']})

    def test_repeated_online_read_does_not_duplicate_steps_or_rewards(self):
        snapshot = {"environment": "shape3d", "task": "shape_identification", "step": 1, "seed": 42, "budget": 20,
                    "views": [{"name": "Agent observation", "image": "test.png"}], "observation": {"instruction": "Inspect shape"},
                    "reward": -.01, "total_reward": -.01, "terminated": False, "truncated": False}
        self.evaluate("SpatialDemo.appendOnline(" + json.dumps(snapshot) + ",true)")
        self.evaluate("SpatialDemo.appendOnline(" + json.dumps(snapshot) + ",false)")
        self.assertEqual(self.js("SpatialDemo.getState().step"), 0)
        self.assertEqual(self.js("document.getElementById('state-budget').textContent"), "19 / 20")
        self.assertEqual(self.js("document.getElementById('state-reward').textContent"), "-0.010")

    def test_research_catalog_has_valid_simulator_and_video_bindings(self):
        ids = {e["id"] for e in self.catalog["environments"]}
        for task in self.catalog["tasks"]:
            self.assertIn(task["environment"], ids)
            self.assertTrue(task["research"]["objective"])
        delivery = next(t for t in self.catalog["tasks"] if t["id"] == "deliverybench")
        self.assertEqual(delivery["environment"], "simworld_delivery")
        for episode in self.catalog["episodes"]:
            if "video" in episode:
                self.assertTrue((OUTPUT / episode["video"]["url"]).is_file())
                self.assertEqual(episode['video']['presentation'],'observation-only')
                self.assertLessEqual(max(episode['video']['width'],episode['video']['height']),640)

    def test_delivery_map_follows_recorded_state(self):
        self.evaluate("Element.prototype.append=function(x){};Element.prototype.replaceChildren=function(){};document.createElement=function(){return new Element({tagName:'DIV',className:'',value:''});};document.createElementNS=function(){return document.createElement();};document.querySelectorAll=function(){return [];};")
        self.evaluate((OUTPUT / "cities.js").read_text())
        self.evaluate((OUTPUT / "static/city.js").read_text())
        route = self.js("CityDemo.shortest([{start:{x:0,y:0},end:{x:1,y:0}},{start:{x:1,y:0},end:{x:1,y:1}}],{x:0,y:0},{x:1,y:1})")
        self.assertEqual(route["distance"], 2)
        self.evaluate("__elements['city-follow'].checked=true;CityDemo.setRecordedState('small-city-11',{position_cm:[100,200],battery_pct:53},[[0,0],[100,200]])")
        actual=json.loads(self.js("document.getElementById('city-recorded-state').textContent"))
        diagram=json.loads(self.js("document.getElementById('city-state').textContent"))
        self.assertEqual(actual['battery_pct'],53)
        self.assertEqual(diagram['diagram_position_m'],{'x':1,'y':2})
        self.evaluate("__elements['city-follow'].checked=false;CityDemo.setRecordedState('small-city-11',{position_cm:[300,400]},[])")
        self.assertEqual(json.loads(self.js("document.getElementById('city-state').textContent"))['diagram_position_m'],{'x':1,'y':2})



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    OUTPUT, RECORD = args.output.resolve(), args.record.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DemoTests))
    report = {"completed": result.wasSuccessful(), "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "javascript_runtime": "QJSEngine with simulated DOM", "full_browser_layout_tested": False,
              "source_rgb_hashes_verified": True if result.wasSuccessful() else "See test failures",
              "notes": "No browser executable or browser-control tool available; no screenshot or responsive-layout claim."}
    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
