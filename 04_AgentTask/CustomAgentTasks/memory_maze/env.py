"""Ordered landmark revisits in native iTHOR; the historical name is retained."""
from copy import deepcopy
from spatial_interaction.environments.thor_tasks.base import ThorTaskEnv
from spatial_interaction.environments.dynamic_memory.interventions import objects, InvalidEpisode
from spatial_interaction.core.actions import InvalidAction


class TaskEnv(ThorTaskEnv):
    @property
    def available_actions(self):
        from spatial_interaction.environments.thor_tasks.base import NAV
        return NAV+(('MEMORIZE',) if self.phase=='exploration' else ('MARK',))+('FINISH',)

    def reset(self,seed=None,options=None):
        self.prepare(seed,options)
        self.phase='exploration';self.phase_steps=0;self.wrong_visits=0;self.discovered=0
        self.explore_budget=int(self.config.task_config.get('exploration_budget',16))
        self.explicit_recall=self.config.task_config.get('memory_protocol')=='explicit_recall_v1'
        self.survey_complete=False
        if self.explicit_recall and not 1 <= self.explore_budget < self.config.max_steps:
            raise ValueError('exploration_budget must leave time for recall')
        types={'Fridge','Microwave','Sink','CoffeeMachine','GarbageCan','StoveBurner','Window'}
        candidates=[o for o in objects(self.controller).values() if o['objectType'] in types and self.public_description(o['objectId']) is not None]
        self.landmarks=[]
        for index in self.rng.permutation(len(candidates)):
            obj=candidates[int(index)]
            try:view=self.view_of(obj['objectId'])
            except InvalidEpisode:continue
            self.landmarks.append((obj['objectId'],view))
            if len(self.landmarks)>=min(4,1+self.config.difficulty):break
        if len(self.landmarks)<2:raise InvalidEpisode('Need two accessible native landmarks')
        self.order=[p[0] for p in self.landmarks]
        if self.config.difficulty>=2:self.order.append(self.order[0])
        self.goal_views={oid:pose for oid,pose in self.landmarks}
        self._teleport(self.initial_pose)
        order=' -> '.join(self.sid(o) for o in self.order)
        self.instruction=('Explore this real indoor scene and remember landmark locations. Use MEMORIZE when ready; '
            'the camera returns to the initial position. Then physically visit and MARK these landmarks in order: '+order+'. '
            'MARK uses {"action":"MARK","object_id":"view:STEP:X:Y"} pointing at the landmark; it must be visible within 1.6 m. '
            'Stay at the final landmark and FINISH. Moves are 0.25 m, turns 90 degrees, look actions 30 degrees. '
            'No map or target coordinates are provided.')
        return self.reset_info()

    def task_action(self,action):
        if action['action']=='MEMORIZE':
            self.memory_checkpoint=action.get('memory',self.memory)
            seen=set()
            for labels,ids in self.visual_frames.values():
                seen.update(oid for index,oid in enumerate(ids) if (labels==index).any())
            self.survey_complete=all(oid in seen for oid,_ in self.landmarks)
            self.phase='recall';self._teleport(self.initial_pose);return
        if action['action']=='MARK':
            obj=self.object_for(action['object_id'])
            if self.progress>=len(self.order) or obj['objectId']!=self.order[self.progress]:
                self.wrong_visits+=1;raise InvalidAction('Landmark is not next in the requested order')
            self.events.append({'event':'landmark_visit','object_id':action['object_id'],'step':self.steps})
            self.progress+=1;self.feedback='Visit recorded. Continue the requested order.'

    def after_action(self):
        if self.phase=='exploration' and not self.done:
            self.phase_steps+=1
            if self.phase_steps>=self.explore_budget and not self.explicit_recall:
                self.phase='recall';self._teleport(self.initial_pose)
                return {'phase_transition':'exploration->recall','reason':'exploration_budget'}

    def evaluate(self):
        obj=objects(self.controller)[self.order[-1]]
        return {'success':self.phase=='recall' and self.progress==len(self.order) and self.planar_distance(obj)<=1.6 and
                    (not self.explicit_recall or self.survey_complete),
                'ordered_visits':self.progress,'required_visits':len(self.order),'wrong_visits':self.wrong_visits,
                'environment_actions':self.steps}

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        if self.explicit_recall and self.phase=='exploration' and self.phase_steps>=self.explore_budget and not self.done:
            self.done=truncated=True
            self.metrics=self.timeout_metrics();self.metrics['failure_reason']='exploration_deadline_without_commit'
            info['metrics']=self.metrics
            self.feedback='Exploration deadline reached without MEMORIZE; recall did not start.'
            obs=self._observation()
        return obs,reward,terminated,truncated,info

    def _observation(self):
        obs=super()._observation()
        if self.explicit_recall:
            if self.phase=='exploration':
                obs['instruction']+=(' Current phase: survey the requested landmarks from useful nearby views and preserve '
                    'their locations in your notebook before MEMORIZE. Notebook updates replace all previous text; '
                    'retain earlier evidence. No automatic commitment. '
                    f'{max(0,self.explore_budget-self.phase_steps)} actions remain to commit.')
            else:
                obs['instruction']+=' Current phase: physically revisit and MARK the landmarks in order; finish at the last one.'
            obs['memory_checkpoint']=self.memory_checkpoint
            obs['task_progress']={'steps_remaining':max(0,self.config.max_steps-self.steps),
                                  'commit_required':self.phase=='exploration'}
        return obs

    def oracle_action(self):
        if self.phase=='exploration':
            if self.discovered>=len(self.landmarks):return {'action':'MEMORIZE'}
            action=self.nav_action(self.landmarks[self.discovered][1])
            if action:return action
            self.discovered+=1
            return {'action':'RotateRight'}
        if self.progress>=len(self.order):return {'action':'FINISH'}
        oid=self.order[self.progress]
        return self.nav_action(self.goal_views[oid]) or {'action':'MARK','object_id':self.sid(oid)}

    def truth(self):
        return {'ordered_landmarks':[self.sid(o) for o in self.order],
                'verified_views':deepcopy(self.goal_views),'scene_semantics':'native indoor landmark revisits; not a generated maze'}
