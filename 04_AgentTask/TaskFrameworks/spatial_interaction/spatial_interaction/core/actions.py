"""Strict JSON action protocol shared by scripted and model agents."""
import json

THOR = ('MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown',
        'OpenObject', 'CloseObject', 'PickupObject')
CAMERA = ('CAMERA_LEFT', 'CAMERA_RIGHT', 'CAMERA_FORWARD', 'CAMERA_BACKWARD',
          'CAMERA_YAW_LEFT', 'CAMERA_YAW_RIGHT', 'CAMERA_PITCH_UP', 'CAMERA_PITCH_DOWN')
ROTATE = tuple(f'OBJECT_ROTATE_{axis}_{sign}' for axis in 'XYZ' for sign in ('POS', 'NEG'))


class InvalidAction(ValueError):
    pass


def parse_action(raw, allowed):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise InvalidAction('Action must be a JSON object') from None
    if not isinstance(raw, dict) or raw.get('action') not in allowed:
        raise InvalidAction('Unknown or unavailable action')
    name = raw['action']
    fields = {'action', 'memory'}
    if name in ('OpenObject', 'CloseObject', 'PickupObject','PutObject','MARK','INSPECT','CLEAN','POWER_ON','PUT_INSIDE'):
        fields.add('object_id')
        if not isinstance(raw.get('object_id'), str):
            raise InvalidAction('object_id is required')
    if name == 'PLACE':
        fields.add('slot_id')
        if not isinstance(raw.get('slot_id'),str):raise InvalidAction('slot_id is required')
    if name == 'SUBMIT':
        fields.add('answer')
        answer = raw.get('answer')
        if not isinstance(answer, dict) or set(answer) != {'changes'} or not isinstance(answer['changes'], list):
            raise InvalidAction('answer must contain a changes list')
        seen = set()
        for change in answer['changes']:
            if not isinstance(change, dict) or change.get('type') not in ('missing', 'moved'):
                raise InvalidAction('Each change needs type missing/moved')
            keys = {'type', 'object_id', 'previous_location'}
            if change['type'] == 'moved':
                keys.add('current_location')
            if set(change) != keys or any(not isinstance(change[k], str) or not change[k] for k in keys):
                raise InvalidAction('Invalid change fields')
            if change['object_id'] in seen:
                raise InvalidAction('Duplicate object_id')
            seen.add(change['object_id'])
    if name == 'SUBMIT_SHAPE':
        fields.add('candidate_id')
        if not isinstance(raw.get('candidate_id'), str):
            raise InvalidAction('candidate_id is required')
    if set(raw) - fields:
        raise InvalidAction('Unexpected action fields')
    if 'memory' in raw and (not isinstance(raw['memory'], str) or len(raw['memory']) > 16000):
        raise InvalidAction('memory must be a string of at most 16000 characters')
    return json.loads(json.dumps(raw))
