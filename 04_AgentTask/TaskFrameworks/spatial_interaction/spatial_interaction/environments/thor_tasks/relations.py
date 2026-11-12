"""Camera-independent metric predicates used by the private state verifier."""
import math


def relation_holds(position,anchor,axes,near=None,margin=.12):
    """axes is a list of (world axis x/z, signed direction); never an action."""
    if not axes or any(axis not in ('x','z') or sign not in (-1,1) for axis,sign in axes):
        raise ValueError('Specify signed world x/z relations')
    directional=all(sign*(position[axis]-anchor[axis])>=margin for axis,sign in axes)
    distance=math.hypot(position['x']-anchor['x'],position['z']-anchor['z'])
    return directional and (near is None or distance<=near)
