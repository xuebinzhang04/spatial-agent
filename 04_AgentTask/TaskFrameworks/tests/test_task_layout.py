"""Directory/task identity, task isolation and moved-entrypoint regression."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

FRAMEWORKS = Path(__file__).resolve().parents[1]
AGENT_TASK = FRAMEWORKS.parent
CUSTOM = AGENT_TASK/'CustomAgentTasks'
NAMES = ('memory_maze','dependency_doors','relational_rearrangement','dynamic_replanning',
         'missing_object','moved_object','shape_identification','pose_alignment')
SPATIAL = NAMES
sys.path.insert(0,str(CUSTOM))


def task_directory(name):
    return CUSTOM/name


def task(name):
    return importlib.import_module(name)


def cli(name, *args):
    return subprocess.run([sys.executable,str(task_directory(name)/'run.py'),*map(str,args)],
                          cwd=AGENT_TASK.parent,capture_output=True,text=True,timeout=45)


class LayoutTests(unittest.TestCase):
    def test_native_and_geometry_tasks_have_independent_entrypoints(self):
        definitions={p.name for p in CUSTOM.iterdir() if p.is_dir() and (p/'task.json').exists()}
        self.assertEqual(definitions,set(NAMES))
        self.assertFalse((CUSTOM/'interaction_lab').exists())
        for name in NAMES:
            folder=task_directory(name)
            self.assertFalse(folder.is_symlink())
            self.assertEqual(json.loads((folder/'task.json').read_text())['name'],name)
            for entry in ('task.py','run.py','README.md','tests'):
                self.assertTrue((folder/entry).exists(),str(folder/entry))

    def test_all_task_apis_bind_identity_and_reject_foreign_specs(self):
        for name in NAMES:
            with self.subTest(task=name):
                module=task(name)
                spec=module.generate()
                self.assertEqual(getattr(spec,'family',getattr(spec,'task_type',None)),name)
                self.assertEqual(spec.to_dict(),module.generate().to_dict())
                other=task(NAMES[(NAMES.index(name)+1)%len(NAMES)]).generate()
                with self.assertRaises(ValueError): module.make_env(other)

    def test_all_cli_help_and_module_entrypoints_work(self):
        for name in NAMES:
            result=cli(name,'--help')
            self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-m','shape_identification.run','--help'],cwd=CUSTOM,
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cli_cannot_select_another_family_or_task(self):
        for name in NAMES:
            result=cli(name,'run','--task','pose_alignment' if name!='pose_alignment' else 'shape_identification','--output','unused')
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('invalid choice',result.stderr)

    def test_foreign_episode_is_rejected_before_simulator_start(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'episode.json'
            path.write_text(json.dumps(task('pose_alignment').generate().to_dict()))
            for verb in ('run','replay'):
                if verb=='run':
                    result=cli('missing_object','run','--episodes',path,'--output',Path(d)/'run')
                else:
                    directory=Path(d)/'episode'; directory.mkdir()
                    (directory/'episode.json').write_text(path.read_text())
                    result=cli('missing_object','replay',directory)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('only',result.stderr)
                self.assertNotIn('Use 02_Simulator',result.stderr)

    def test_filtered_generation_and_benchmarks_never_mix_tasks(self):
        for name in SPATIAL:
            module=task(name)
            from spatial_interaction.benchmarks import generate_episodes
            configs=generate_episodes(5,task_type=name)
            self.assertEqual(len(configs),5)
            self.assertEqual({c.task_type for c in configs},{name})
            self.assertEqual(len({c.episode_id for c in configs}),5)
            for split in ('development','validation','test'):
                rows=[json.loads(x) for x in (CUSTOM/name/'benchmarks'/f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual({r['task_type'] for r in rows},{name})

    def test_moved_demo_resolves_existing_workspace_assets(self):
        demo=AGENT_TASK/'Demos/spatial_agent'
        spec=importlib.util.spec_from_file_location('_layout_delivery_adapter',demo/'delivery.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(module.ROOT,AGENT_TASK.parent)
        self.assertTrue((module.TASK/'maps/small-city-11/roads.json').is_file())
        self.assertTrue((demo/'index.html').is_file())
        self.assertTrue((demo/'run.py').is_file())


if __name__=='__main__':
    unittest.main(verbosity=2)
