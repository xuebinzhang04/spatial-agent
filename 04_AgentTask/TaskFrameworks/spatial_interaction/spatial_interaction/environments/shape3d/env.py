from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from ...core.base_env import SpatialEnv
from ...core.actions import CAMERA, ROTATE, InvalidAction
from .geometry import (generate_shape, canonical, ambiguous_distractor, occluded_pair, ROTATIONS,
                       quarter_turn, rotation_error, shortest_alignment, shape_split, symmetry_group)
from .renderer import render, camera_basis


class Shape3DEnv(SpatialEnv):
    @property
    def available_actions(self):
        if self.config.task_type == 'shape_identification':
            return CAMERA + ('SUBMIT_SHAPE',)
        return CAMERA + tuple(a for a in ROTATE if a.split('_')[2] in self.axes) + ('SUBMIT_POSE',)

    def reset(self, seed=None, options=None):
        self._start(seed, options)
        cfg = self.config.task_config
        self.size = int(cfg.get('image_size', 224))
        if not 64 <= self.size <= 512:
            raise ValueError('image_size must be 64..512')
        self.axes = cfg.get('axes', 'XYZ')
        if not self.axes or set(self.axes) - set('XYZ'):
            raise ValueError('axes must be a nonempty subset of XYZ')
        self.cells = generate_shape(self.rng, int(cfg.get('voxels', 5+self.config.difficulty)),
                                    cfg.get('symmetry', 'asymmetric'), cfg.get('split'))
        self.require_ambiguity=bool(cfg.get('require_initial_pixel_ambiguity',False))
        hidden_alternative=None
        if self.require_ambiguity:
            if self.config.task_type!='shape_identification':
                raise ValueError('Initial pixel ambiguity is only supported for shape identification')
            self.cells,hidden_alternative=occluded_pair(self.rng,int(cfg.get('voxels',6)),cfg.get('split'))
        span = float(np.ptp(self.cells, axis=0).max()+1)
        self.distance = max(7., span*1.7)
        self.camera = np.array(cfg.get('camera_position', [0., .2, -self.distance]), dtype=float)
        if self.camera.shape != (3,) or not np.isfinite(self.camera).all() or np.linalg.norm(self.camera) < span:
            raise ValueError('camera_position must be a finite exterior 3D position')
        self.yaw = float(cfg.get('camera_yaw', 0.))
        self.pitch = float(cfg.get('camera_pitch', 0.))
        initial_rotation = cfg.get('initial_rotation', int(self.rng.integers(24)) if self.config.difficulty > 1 else 0)
        if type(initial_rotation) is not int or not 0 <= initial_rotation < 24:
            raise ValueError('initial_rotation must index the 24 proper cube rotations')
        self.orientation = ROTATIONS[initial_rotation].copy()
        self.reference = ROTATIONS[int(self.rng.integers(24))].copy()
        self.alignment_length = None
        if self.config.task_type == 'pose_alignment':
            self.reference, self.alignment_length = self._sample_reference(
                cfg.get('alignment_length', self.config.difficulty))
        self.occlusion = float(cfg.get('occlusion', .35 if self.config.difficulty >= 4 else 0.))
        if not 0 <= self.occlusion <= 1.:
            raise ValueError('occlusion must be 0..1')
        self.camera_actions = self.object_rotations = 0
        self.viewpoints = {self._view_key()}
        self.candidates = [self.cells.copy()]
        self.ambiguous = False
        count = int(cfg.get('candidates', min(5, 2+self.config.difficulty)))
        if not 2 <= count <= 6:
            raise ValueError('candidates must be 2..6')
        if self.config.task_type == 'shape_identification':
            keys = {canonical(self.cells)}
            ambiguous = hidden_alternative if hidden_alternative is not None else ambiguous_distractor(self.cells)
            if ambiguous is not None and (not cfg.get('split') or shape_split(ambiguous) == cfg['split']):
                self.candidates.append(ambiguous)
                keys.add(canonical(ambiguous))
                self.ambiguous = True
            while len(self.candidates) < count:
                cells = generate_shape(self.rng, len(self.cells), 'any', cfg.get('split'), keys)
                keys.add(canonical(cells))
                self.candidates.append(cells)
            order = self.rng.permutation(count)
            self.candidates = [self.candidates[int(i)] for i in order]
            self.correct = f'candidate_{list(order).index(0)+1}'
        else:
            self.correct = None
        self.instruction = self.config.instruction or (
            'Identify the target shape. Candidate panels show three fixed views each. '
            'Explore with CAMERA actions, then submit {"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}.'
            if self.config.task_type == 'shape_identification' else
            'Rotate the target to match the reference panel using 90 degree world-axis rotations. '
            'Camera motion does not rotate the object. Submit {"action":"SUBMIT_POSE"} when aligned.')
        self.ambiguity_witness=None
        if self.require_ambiguity:
            initial=render(self.cells,self.orientation,self.camera,self.yaw,self.pitch,self.size*2,self.occlusion)
            twin=render(hidden_alternative,self.orientation,self.camera,self.yaw,self.pitch,self.size*2,self.occlusion)
            if not np.array_equal(initial,twin):
                raise ValueError('Candidate pair is distinguishable in actual initial pixels; reject this case')
            right,_,forward=camera_basis(self.yaw,0.)
            # Search sides in a seed-dependent order (separate stream, so episode
            # generation is unchanged) to avoid a systematically left-first witness.
            sides=[int(x) for x in np.random.default_rng([self.config.seed,7]).permutation([-1,1])]
            for count in range(1,min(11,self.config.max_steps-1)):
                for sign in sides:
                    side=self.camera+sign*.6*count*right
                    forward_steps=max(0,int(round((self.distance-np.sqrt(max(0,self.distance**2-(.6*count)**2)))/.6)))
                    camera=side+.6*forward_steps*forward
                    bound=np.linalg.norm(np.ptp(self.cells,axis=0)+1)/2+.8
                    if not bound<=np.linalg.norm(camera)<=self.distance*2:continue
                    desired=np.degrees(np.arctan2(-camera[0],-camera[2]))
                    turns=int(round(((desired-self.yaw+180)%360-180)/15))
                    yaw=self.yaw+15*turns
                    if count+forward_steps+abs(turns)+1>self.config.max_steps:continue
                    if any(not bound<=np.linalg.norm(self.camera+sign*.6*i*right)<=self.distance*2 for i in range(1,count+1)):continue
                    if any(not bound<=np.linalg.norm(side+.6*i*forward)<=self.distance*2 for i in range(1,forward_steps+1)):continue
                    a=render(self.cells,self.orientation,camera,yaw,self.pitch,self.size*2,self.occlusion)
                    b=render(hidden_alternative,self.orientation,camera,yaw,self.pitch,self.size*2,self.occlusion)
                    occupied=np.any(a!=[27,35,51],axis=2)|np.any(b!=[27,35,51],axis=2)
                    changed=np.max(np.abs(a.astype(int)-b.astype(int)),axis=2)>10
                    fraction=float(np.count_nonzero(changed&occupied)/max(1,np.count_nonzero(occupied)))
                    retained=float(np.count_nonzero(np.any(a!=[27,35,51],axis=2))/max(1,np.count_nonzero(np.any(initial!=[27,35,51],axis=2))))
                    if fraction>=.10 and retained>=.60:
                        actions=([{'action':'CAMERA_RIGHT' if sign>0 else 'CAMERA_LEFT'}]*count+
                                 [{'action':'CAMERA_FORWARD'}]*forward_steps+
                                 [{'action':'CAMERA_YAW_RIGHT' if turns>0 else 'CAMERA_YAW_LEFT'}]*abs(turns))
                        self.ambiguity_witness={'actions':actions,'changed_foreground_fraction':fraction,
                                                'retained_target_area_ratio':retained}
                        break
                if self.ambiguity_witness:break
            if not self.ambiguity_witness:
                raise ValueError('No bounded camera translation resolves the initial ambiguity')
            # State that the first view may be insufficient, but not which camera action resolves it.
            self.instruction+=(' The initial target view may not distinguish every candidate. '
                'Gather the evidence you need before submitting. Preserve observations in optional memory notes.')
        self.panel = self._reference_panel()
        witness = shortest_alignment(self.orientation, self.reference, self.cells, self.axes)
        if self.config.task_type == 'pose_alignment' and len(witness)+1 > self.config.max_steps:
            raise ValueError('Action budget cannot contain a solution')
        return self._observation(), {'valid': True, 'ground_truth': self.ground_truth()}

    def _sample_reference(self, requested):
        """Sample a reference pose exactly `requested` quarter turns away (symmetry-aware).

        Random walks of quarter turns cancel out (the cube group's diameter is 3), so
        difficulty would not control rotation depth. Instead, breadth-first distances
        over the allowed axes define each pose's depth; requests beyond the deepest
        reachable pose are capped to it. Returns (reference, depth).
        """
        if type(requested) is not int or requested < 1:
            raise ValueError('alignment_length must be a positive integer')
        distance = {tuple(self.orientation.flat): 0}
        frontier = [self.orientation]
        while frontier:
            following = []
            for rotation in frontier:
                for axis in self.axes:
                    for sign in (1, -1):
                        new = quarter_turn(axis, sign) @ rotation
                        if tuple(new.flat) not in distance:
                            distance[tuple(new.flat)] = distance[tuple(rotation.flat)] + 1
                            following.append(new)
            frontier = following
        # A reference R is aligned at every pose R @ s for a shape symmetry s.
        symmetries = symmetry_group(self.cells)
        by_depth = {}
        for rotation in ROTATIONS:
            depths = [distance[k] for k in (tuple((rotation @ s).flat) for s in symmetries) if k in distance]
            if depths and min(depths) > 0:
                by_depth.setdefault(min(depths), []).append(rotation)
        depth = min(requested, max(by_depth))
        options = by_depth[depth]
        return options[int(self.rng.integers(len(options)))].copy(), depth

    def _view_key(self):
        return tuple(np.round(self.camera, 6)) + (round(self.yaw, 6), round(self.pitch, 6))

    def _reference_panel(self):
        width = self.size
        canvas = Image.new('RGB', (width, self.size*2), (18, 23, 34))
        draw = ImageDraw.Draw(canvas)
        if self.config.task_type == 'pose_alignment':
            reference = render(self.cells, self.reference, [0,.2,-self.distance], size=self.size)
            canvas.paste(Image.fromarray(reference), (0, 25))
            draw.text((5, 6), 'REFERENCE (fixed camera)', fill='white')
        else:
            row_height = min(100, (self.size*2)//len(self.candidates))
            thumb = min((width-10)//3, row_height-22)
            for i, cells in enumerate(self.candidates):
                top = i*row_height
                draw.text((5, top+2), f'candidate_{i+1}', fill='white')
                for j, rotation in enumerate((np.eye(3), quarter_turn('Y'), quarter_turn('X'))):
                    image = render(cells, rotation, [0,.2,-self.distance], size=thumb)
                    canvas.paste(Image.fromarray(image), (5+j*thumb, top+20))
        return canvas

    def _execute(self, action):
        name = action['action']
        if name == 'SUBMIT_SHAPE':
            if action['candidate_id'] not in [f'candidate_{i+1}' for i in range(len(self.candidates))]:
                raise InvalidAction('Unknown candidate_id')
            success = action['candidate_id'] == self.correct
            self.metrics = self._metrics(success)
            self.metrics['shape_accuracy'] = float(success)
            return float(success), {'metrics': self.metrics}
        if name == 'SUBMIT_POSE':
            error = rotation_error(self.orientation, self.reference, self.cells)
            self.metrics = self._metrics(error < 1e-6)
            self.metrics['rotation_error_degrees'] = error
            return float(error < 1e-6), {'metrics': self.metrics}
        if name.startswith('OBJECT_ROTATE_'):
            _, _, axis, sign = name.split('_')
            self.orientation = quarter_turn(axis, 1 if sign == 'POS' else -1) @ self.orientation
            self.object_rotations += 1
        else:
            right, _, forward = camera_basis(self.yaw, 0.)
            translations = {'CAMERA_LEFT': -right, 'CAMERA_RIGHT': right,
                            'CAMERA_FORWARD': forward, 'CAMERA_BACKWARD': -forward}
            if name in translations:
                new = self.camera + .6*translations[name]
                bound = np.linalg.norm(np.ptp(self.cells, axis=0)+1)/2 + .8
                if np.linalg.norm(new) < bound or np.linalg.norm(new) > self.distance*2:
                    raise InvalidAction('Camera translation outside safe viewing volume')
                self.camera = new
            elif name == 'CAMERA_YAW_LEFT': self.yaw -= 15.
            elif name == 'CAMERA_YAW_RIGHT': self.yaw += 15.
            elif name == 'CAMERA_PITCH_UP': self.pitch = min(65., self.pitch+10.)
            elif name == 'CAMERA_PITCH_DOWN': self.pitch = max(-65., self.pitch-10.)
            self.camera_actions += 1
            self.viewpoints.add(self._view_key())
        return None, {}

    def _metrics(self, success):
        return {'success': bool(success), 'camera_actions': self.camera_actions,
                'object_rotations': self.object_rotations, 'total_steps': self.steps,
                'environment_actions': self.steps, 'viewpoints_used': len(self.viewpoints),
                'occlusion': self.occlusion}

    def timeout_metrics(self):
        return self._metrics(False)

    def render(self):
        target = render(self.cells, self.orientation, self.camera, self.yaw, self.pitch,
                        self.size*2, self.occlusion)
        canvas = Image.new('RGB', (self.size*3, self.size*2))
        canvas.paste(Image.fromarray(target), (0,0))
        canvas.paste(self.panel, (self.size*2,0))
        ImageDraw.Draw(canvas).text((8,8), 'TARGET (agent camera)', fill='white')
        return np.asarray(canvas, dtype=np.uint8).copy()

    def ground_truth(self):
        return {'candidate_id': self.correct, 'reference_orientation': self.reference.tolist(),
                'shape': self.cells.tolist(), 'ambiguous_front_silhouette': self.ambiguous}

    def privileged_state(self):
        return {'ground_truth': self.ground_truth(), 'orientation': self.orientation.tolist(),
                'camera_position': self.camera.tolist(), 'camera_yaw': self.yaw,
                'camera_pitch': self.pitch, 'ambiguity_witness':deepcopy(self.ambiguity_witness), 'candidate_geometry': [c.tolist() for c in self.candidates]}
