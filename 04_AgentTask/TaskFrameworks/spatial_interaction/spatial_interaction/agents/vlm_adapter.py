"""Thin adapter to the repository's existing OpenAI-compatible Qwen/vLLM server.

Input modes (``history``):
  * ``'rollout'`` (default): multi-turn chat in the style of Theory of Space / VAGEN.
    Every past turn stays in context: the model's raw replies and, for each turn, the
    execution feedback plus the single image shown after that turn. ``image_window=K``
    keeps images only in the K most recent observation messages (older turns keep
    their text); ``None`` keeps every image, as Theory of Space does.
  * ``'memory'``: the earlier stateless input. One message with the current image,
    action history and the agent-authored notebook. Kept as the belief-bottleneck
    alternative.

Each turn the model may return up to ``max_actions`` actions (VAGEN-style). The
caller executes them in order, stops at the first failed action or episode end, and
presents only the image after the last executed action.

No model loading, training, evaluator info, or environment object enters this class.
"""
import base64
import io
import json
import os
import re
from copy import deepcopy
from urllib.request import Request, urlopen
from PIL import Image
from ..core.actions import parse_action, InvalidAction

PUBLIC_FIELDS = ('instruction', 'phase', 'action_history', 'memory', 'feedback', 'view_step',
                 'memory_checkpoint', 'task_progress')
# Per-turn state sent in rollout mode; the instruction is re-sent only when the phase changes.
TURN_FIELDS = ('phase', 'view_step', 'task_progress', 'memory_checkpoint', 'feedback')

ACTION_RULES = (
    'Navigation/rotation: {"action":"NAME"}. Object references are strings "view:STEP:X:Y": STEP is an '
    'observed view_step and X,Y are normalized [0,1) image coordinates from the top-left on the object\'s '
    'visible pixels. Interactions add "object_id" as such a reference. SUBMIT requires '
    '"answer":{"changes":[{"type":"missing","object_id":"view:STEP:X:Y","previous_location":"view:STEP:X:Y"}]}; '
    'moved changes also require "current_location". SUBMIT_SHAPE requires "candidate_id":"candidate_N". '
    'SUBMIT_POSE has no additional required fields. Native migrated tasks use FINISH to submit their physical '
    'state. MARK/INSPECT/CLEAN/POWER_ON/PUT_INSIDE require an object_id reference; PLACE requires slot_id, read '
    'from visible slot markers. Optional "memory" on an action replaces your complete notebook. '
    'Follow the task instruction.')


def turn_format(max_actions):
    return (f'Reply with a JSON list of 1 to {max_actions} action objects, executed in order, for example '
            '[{"action":"RotateLeft"},{"action":"MoveAhead"}]. A single action object is also accepted; no prose. '
            'Execution stops at the first failed action or when the episode ends, and you then see only the '
            'image after the last executed action.')


def png_url(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def describe_results(results):
    """Public text for what happened to each action in the previous turn."""
    lines = ['Executed actions:']
    for index, result in enumerate(r for r in results if 'action' in r):
        status = 'ok' if result['valid'] else 'failed: ' + str(result.get('error'))
        if result.get('phase_transition'):
            status += f" (phase {result['phase_transition']})"
        lines.append(f"{index + 1}. {json.dumps(result['action'])} -> {status}")
    skipped = sum(r.get('skipped', 0) for r in results)
    if skipped:
        lines.append(f'{skipped} later action(s) were not executed.')
    return '\n'.join(lines)


class VLMAdapter:
    def __init__(self, base_url=None, model=None, transport=None, timeout=120,
                 history='rollout', max_actions=3, image_window=None):
        if history not in ('rollout', 'memory'):
            raise ValueError("history must be 'rollout' or 'memory'")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError('max_actions must be a positive integer')
        if image_window is not None and (type(image_window) is not int or image_window < 1):
            raise ValueError('image_window must be None or a positive integer')
        self.base_url = (base_url or os.environ.get('QWEN_BASE_URL', 'http://127.0.0.1:8000/v1')).rstrip('/')
        self.model = model or os.environ.get('QWEN_MODEL', 'Qwen3.8-27B')
        self.timeout = timeout
        self.transport = transport or self._request
        self.history, self.max_actions, self.image_window = history, max_actions, image_window
        self.calls = []
        self.messages = []  # rollout-mode conversation, images stored as data URLs
        self.last_phase = None

    def _request(self, payload):
        request = Request(self.base_url+'/chat/completions', data=json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json', 'Authorization':'Bearer local-unused'})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _complete(self, messages):
        payload = {'model':self.model, 'temperature':0., 'max_tokens':1024,
                   'chat_template_kwargs':{'enable_thinking':False}, 'messages':messages}
        try:
            response = self.transport(payload)
        except Exception as error:
            self.calls.append({'transport_error': f'{type(error).__name__}: {error}'})
            raise
        return response['choices'][0]['message'].get('content') or ''

    @staticmethod
    def _clean(text):
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        return re.sub(r'^```(?:json)?\s*|\s*```$', '', clean).strip()

    # ---- legacy single-action interface (memory mode, one action per call) ----
    def act(self, observation, available_actions):
        public = {k:observation.get(k) for k in PUBLIC_FIELDS}
        prompt = ('Choose exactly one action as a JSON object; no prose. Allowed actions: '
                  + ', '.join(available_actions) + '. ' + ACTION_RULES
                  + ' Preserve previous observations when updating the notebook.\n' + json.dumps(public))
        text = self._complete([{'role':'user', 'content':[{'type':'text','text':prompt},
                                {'type':'image_url','image_url':{'url':png_url(observation['rgb'])}}]}])
        try:
            action = parse_action(self._clean(text), available_actions)
            error = None
        except InvalidAction as exc:
            # Invalid model responses consume an ordinary environment step.
            action, error = {'action':'INVALID'}, str(exc)
        self.calls.append({'response':text, 'action':action, 'parse_error':error})
        return action

    # ---- turn interface: up to max_actions actions per model call ----
    def parse_turn(self, text):
        """Return (actions, error, truncated). Field validity is checked by the environment."""
        try:
            value = json.loads(self._clean(text))
        except (json.JSONDecodeError, TypeError):
            return [{'action':'INVALID'}], 'Reply must be a JSON action object or list', False
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not value or not all(isinstance(a, dict) for a in value):
            return [{'action':'INVALID'}], 'Reply must be a nonempty list of action objects', False
        return value[:self.max_actions], None, len(value) > self.max_actions

    def act_turn(self, observation, available_actions, results=None):
        """Choose this turn's actions from the latest observation and previous turn results."""
        messages = (self._rollout_messages(observation, available_actions, results) if self.history == 'rollout'
                    else self._memory_messages(observation, available_actions, results))
        text = self._complete(messages)
        actions, error, truncated = self.parse_turn(text)
        if self.history == 'rollout':
            self.messages.append({'role':'assistant', 'content':text})
        self.calls.append({'turn':len(self.calls), 'history':self.history, 'response':text, 'actions':actions,
                           'parse_error':error, 'truncated':truncated, 'messages_in_request':len(messages),
                           'images_in_request':sum(part.get('type') == 'image_url' for m in messages
                                                   if isinstance(m['content'], list) for part in m['content'])})
        return actions

    def _rollout_messages(self, observation, available_actions, results):
        if not self.messages:
            self.messages.append({'role':'system', 'content':(
                'You control an agent in a visual spatial environment through multi-turn interaction. '
                + ACTION_RULES + ' ' + turn_format(self.max_actions))})
        text = []
        if results:
            text.append(describe_results(results))
        if observation.get('phase') != self.last_phase:
            # The instruction carries the task and the stage text; re-send it when the stage changes.
            text.append('Task instruction: ' + str(observation.get('instruction')))
            self.last_phase = observation.get('phase')
        state = {k:observation[k] for k in TURN_FIELDS if observation.get(k) not in (None, '')}
        text.append('Current state: ' + json.dumps(state))
        text.append('Allowed actions now: ' + ', '.join(available_actions) + '.')
        step = observation.get('view_step')
        text.append('Observation:' if step is None else f'Observation (view_step {step}):')
        self.messages.append({'role':'user', 'content':[{'type':'text', 'text':'\n'.join(text)},
                              {'type':'image_url', 'image_url':{'url':png_url(observation['rgb'])}}]})
        return self._windowed(self.messages)

    def _windowed(self, messages):
        """Drop images outside the most recent image_window observation messages."""
        if self.image_window is None:
            return list(messages)  # the request must not alias the growing conversation
        messages = deepcopy(messages)
        kept = 0
        for message in reversed(messages):
            if not isinstance(message['content'], list):
                continue
            if any(part.get('type') == 'image_url' for part in message['content']):
                kept += 1
                if kept > self.image_window:
                    message['content'] = [part if part.get('type') != 'image_url' else
                                          {'type':'text', 'text':'[earlier image omitted]'} for part in message['content']]
        return messages

    def _memory_messages(self, observation, available_actions, results):
        public = {k:observation.get(k) for k in PUBLIC_FIELDS}
        if results:
            public['last_turn'] = describe_results(results)
        prompt = ('Allowed actions: ' + ', '.join(available_actions) + '. ' + ACTION_RULES + ' '
                  + turn_format(self.max_actions)
                  + ' Earlier images are not replayed; preserve observations in the notebook.\n' + json.dumps(public))
        return [{'role':'user', 'content':[{'type':'text','text':prompt},
                 {'type':'image_url','image_url':{'url':png_url(observation['rgb'])}}]}]
