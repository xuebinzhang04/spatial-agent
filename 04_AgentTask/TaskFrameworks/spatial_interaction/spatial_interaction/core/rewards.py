"""Pure evaluators: no model or simulator dependency."""


def evaluate_changes(answer, truth, steps=0, max_steps=50):
    expected = {c['object_id']: c for c in truth['changes']}
    predicted = {c['object_id']: c for c in answer.get('changes', [])}
    common = set(expected) & set(predicted)
    precision = len(common) / len(predicted) if predicted else 0.0
    recall = len(common) / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    def accuracy(field, ids):
        return sum(predicted.get(i, {}).get(field) == expected[i][field] for i in ids) / len(ids) if ids else 1.0
    moved = [i for i, c in expected.items() if c['type'] == 'moved']
    type_acc = accuracy('type', list(expected))
    previous = accuracy('previous_location', list(expected))
    current = accuracy('current_location', moved)
    return {'change_precision': precision, 'change_recall': recall, 'change_f1': f1,
            'change_type_accuracy': type_acc, 'previous_location_accuracy': previous,
            'current_location_accuracy': current,
            'success': f1 == type_acc == previous == current == 1.0,
            'environment_actions': steps, 'exploration_efficiency': f1 / max(steps, 1),
            'episode_reward': f1 - 0.1 * steps / max_steps}
