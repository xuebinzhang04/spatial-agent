"""Headless perspective ray/voxel renderer with exact depth occlusion; NumPy only."""
import numpy as np


def camera_basis(yaw, pitch):
    yaw, pitch = np.radians([yaw, pitch])
    forward = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    right = np.array([np.cos(yaw), 0., -np.sin(yaw)])
    up = np.cross(forward, right)
    return right, up, forward


def render(cells, orientation, camera, yaw=0., pitch=0., size=224, occlusion=0.):
    right, up, forward = camera_basis(yaw, pitch)
    grid = (np.arange(size) + .5) / size * 2 - 1
    xx, yy = np.meshgrid(grid, -grid)
    rays = forward + (xx[..., None]*right + yy[..., None]*up)*np.tan(np.radians(50)/2)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    origin = np.asarray(camera, dtype=float)
    cells = np.asarray(cells, dtype=float)
    center = (cells.min(axis=0) + cells.max(axis=0))/2
    boxes = cells - center
    local_origin, local_rays = origin @ orientation, rays @ orientation
    depth = np.full((size, size), np.inf)
    shade = np.zeros((size, size))
    edges = np.zeros((size, size), dtype=bool)
    light = np.array([-.4, .8, -1.])
    light /= np.linalg.norm(light)
    # Slab intersection uses a finite parallel-ray reciprocal; origins outside
    # parallel slabs have no intersection, inside slabs leave the other axes free.
    inverse = np.divide(1., local_rays, out=np.full_like(local_rays, 1e12), where=np.abs(local_rays)>1e-12)
    for box in boxes:
        lo, hi = box-.5, box+.5
        t1, t2 = (lo-local_origin)*inverse, (hi-local_origin)*inverse
        entry_axis = np.argmax(np.minimum(t1, t2), axis=-1)
        near, far = np.min(np.maximum(t1, t2), axis=-1), np.max(np.minimum(t1, t2), axis=-1)
        hit = (near >= np.maximum(far, .05)) & (far < depth) & (far > .05)
        normal = np.eye(3)[entry_axis] * -np.take_along_axis(np.sign(local_rays), entry_axis[..., None], axis=-1)
        illumination = .35 + .65*np.maximum(0., (normal @ orientation.T) @ light)
        position = local_origin + far[..., None]*local_rays - box
        border = np.sort(np.abs(position), axis=-1)[..., 1] > .47
        depth[hit] = far[hit]
        shade[hit] = illumination[hit]
        edges[hit] = border[hit]
    background = np.broadcast_to(np.array([27, 35, 51], dtype=float), (size,size,3)).copy()
    mask = np.isfinite(depth)
    color = np.array([83., 193., 220.])
    background[mask] = color*shade[mask, None]
    background[edges & mask] *= .65
    if occlusion:
        # Physical vertical foreground screen at world z=-2.5, x width set by difficulty.
        t = (-2.5-origin[2]) / np.where(np.abs(rays[...,2])>1e-10, rays[...,2], 1e-10)
        point = origin + t[...,None]*rays
        screen = (t > 0) & (t < depth) & (np.abs(point[...,0]) < occlusion) & (np.abs(point[...,1]) < 2.)
        background[screen] = [113, 105, 91]
    return np.clip(background, 0, 255).astype(np.uint8)
