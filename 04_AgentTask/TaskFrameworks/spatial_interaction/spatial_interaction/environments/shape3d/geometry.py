"""Integer polycubes and the 24 proper cube rotations (no reflections)."""
from collections import deque
import hashlib
import numpy as np


def quarter_turn(axis, sign=1):
    matrices = [np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]
    rotation = matrices['XYZ'.index(axis)]
    return rotation if sign > 0 else rotation.T


def cube_rotations():
    rotations = [np.eye(3, dtype=int)]
    seen = {tuple(rotations[0].flat)}
    for rotation in rotations:
        for axis in 'XYZ':
            new = quarter_turn(axis) @ rotation
            key = tuple(new.flat)
            if key not in seen:
                seen.add(key)
                rotations.append(new)
    return rotations


ROTATIONS = cube_rotations()


def normalized(cells):
    cells = np.asarray(cells, dtype=int)
    return tuple(sorted(map(tuple, (cells - cells.min(axis=0)).tolist())))


def canonical(cells):
    cells = np.asarray(cells, dtype=int)
    return min(normalized(cells @ r.T) for r in ROTATIONS)


def symmetry_group(cells):
    base = normalized(cells)
    return [r for r in ROTATIONS if normalized(np.asarray(cells) @ r.T) == base]


def rotation_error(current, reference, cells):
    errors = []
    for symmetry in symmetry_group(cells):
        relative = current.T @ reference @ symmetry
        errors.append(float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))))
    return min(errors)


def shortest_alignment(current, reference, cells, axes='XYZ'):
    queue = deque([(current, [])])
    seen = {tuple(current.flat)}
    while queue:
        rotation, path = queue.popleft()
        if rotation_error(rotation, reference, cells) < 1e-6:
            return path
        for axis in axes:
            for sign, word in ((1, 'POS'), (-1, 'NEG')):
                new = quarter_turn(axis, sign) @ rotation
                key = tuple(new.flat)
                if key not in seen:
                    seen.add(key)
                    queue.append((new, path + [f'OBJECT_ROTATE_{axis}_{word}']))
    raise ValueError('Reference is unreachable with configured axes')


def shape_split(cells):
    digest = hashlib.sha256(repr(canonical(cells)).encode()).hexdigest()
    return ('development', 'validation', 'test')[int(digest[:12], 16) % 3]


def generate_shape(rng, voxels=6, symmetry='asymmetric', split=None, excluded=()):
    if not 4 <= voxels <= 30:
        raise ValueError('voxels must be 4..30')
    if symmetry not in ('asymmetric', 'symmetric', 'any'):
        raise ValueError('symmetry must be asymmetric, symmetric, or any')
    if split not in (None, 'development', 'validation', 'test'):
        raise ValueError('Unknown shape split')
    for _ in range(5000):
        cells = {(0, 0, 0)}
        if symmetry == 'symmetric' and voxels % 2 == 0:
            cells.add((1, 0, 0))
        while len(cells) < voxels:
            cell = np.array(sorted(cells)[int(rng.integers(len(cells)))])
            axis = int(rng.integers(3))
            cell[axis] += int(rng.choice([-1, 1]))
            proposed = {tuple(cell)}
            if symmetry == 'symmetric':
                # Half-turn symmetry around world Y; even counts use x=.5.
                proposed.add((int(voxels % 2 == 0)-cell[0], cell[1], -cell[2]))
            if len(cells | proposed) <= voxels:
                cells |= proposed
        cells = np.asarray(normalized(list(cells)), dtype=int)
        key = canonical(cells)
        if key in excluded or (symmetry == 'asymmetric' and len(symmetry_group(cells)) != 1):
            continue
        if split and shape_split(cells) != split:
            continue
        return cells
    raise ValueError('Unable to generate requested unique shape/split')


def ambiguous_distractor(cells, excluded=()):
    """Try connected equal-count shapes with the same orthographic front silhouette.

    Perspective images are not claimed to be identical.
    """
    cells = np.asarray(cells, dtype=int)
    silhouette = set(map(tuple, cells[:, :2]))
    for index, cell in enumerate(cells):
        for dz in (-2, -1, 1, 2):
            new = cells.copy()
            new[index, 2] += dz
            points = set(map(tuple, new))
            if len(points) != len(cells) or set(map(tuple, new[:, :2])) != silhouette:
                continue
            seen = {next(iter(points))}
            pending = list(seen)
            for point in pending:
                for other in sorted(points - seen):
                    if sum(abs(a-b) for a, b in zip(point, other)) == 1:
                        seen.add(other)
                        pending.append(other)
            if len(seen) == len(points) and canonical(new) != canonical(cells) and canonical(new) not in excluded:
                return np.asarray(normalized(new))
    return None


def occluded_pair(rng, voxels=6, split=None):
    """Connected shapes sharing a front layer but differing behind it.

    The caller must still verify actual perspective pixels and a revealing view.
    Geometry alone is not an observation ambiguity certificate.
    """
    import itertools
    front_count=voxels-2
    if not 4 <= front_count <= 10:
        raise ValueError('Occluded pairs support 6..12 voxels')
    for _ in range(500):
        front={(0,0,0)}
        while len(front)<front_count:
            cell=list(sorted(front)[int(rng.integers(len(front)))])
            axis=int(rng.integers(2));cell[axis]+=int(rng.choice([-1,1]))
            front.add(tuple(cell))
        positions=sorted(front)
        pairs=[]
        keys=set()
        for chosen in itertools.combinations(positions,2):
            cells=np.asarray(normalized(list(front)+[(x,y,1) for x,y,_ in chosen]))
            key=canonical(cells)
            if key in keys or split and shape_split(cells)!=split:continue
            keys.add(key);pairs.append(cells)
        if len(pairs)>=2:
            order=rng.permutation(len(pairs))
            return pairs[int(order[0])],pairs[int(order[1])]
    raise ValueError('No distinct occluded pair in the requested split')
