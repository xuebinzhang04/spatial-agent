"""Offline-only cached THOR launcher; never downloads/prunes simulator builds."""
from pathlib import Path
import os
import time


def workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / '02_Simulator/AI2THOR/runtime').is_dir():
            return parent
    raise RuntimeError('Cannot locate existing SpatialAgent simulator bundle')


def create_controller(scene, size, runtime_dir, depth=False):
    try:
        from ai2thor.controller import Controller
        from ai2thor.build import Build, COMMIT_ID
        from ai2thor.platform import CloudRendering
        from ai2thor.fifo_server import FifoServer
    except ImportError as error:
        raise RuntimeError('Use 02_Simulator/AI2THOR/runtime/python/bin/python for THOR tasks') from error
    runtime = workspace_root() / '02_Simulator/AI2THOR/runtime'
    build = Build(CloudRendering, COMMIT_ID, False, str(runtime / 'releases'))
    if not Path(build.executable_path).is_file():
        raise RuntimeError(f'Matching cached build absent: {build.executable_path}; automatic download disabled')
    runtime_dir = Path(runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    player_log = runtime_dir / f'player-{time.time_ns()}.log'

    class CachedController(Controller):
        @property
        def base_dir(self):
            return str(runtime_dir)

        def find_build(self, *args):
            return build

        def prune_releases(self):
            pass

        def unity_command(self, *args):
            command = super().unity_command(*args)
            # Explicit device selection permits lavapipe; do not set gpu_device,
            # which invokes CUDA-to-Vulkan mapping and requires nvidia-smi.
            if os.environ.get('SPATIAL_THOR_SOFTWARE') == '1':
                command += ['-force-device-index', '0']
            return command + ['-logFile', str(player_log)]

    controller = CachedController.__new__(CachedController)
    try:
        controller.__init__(scene=scene, server_class=FifoServer, width=size, height=size,
                            quality='Low', gridSize=.25, rotateStepDegrees=30, snapToGrid=False,
                            antiAliasing='none',
                            visibilityDistance=5., renderInstanceSegmentation=True,renderDepthImage=depth,
                            server_start_timeout=35., server_timeout=20.)
    except Exception:
        if getattr(controller, 'server', None):
            controller.stop()
        raise
    return controller
