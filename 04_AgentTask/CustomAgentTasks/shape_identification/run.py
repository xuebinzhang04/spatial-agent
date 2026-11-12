#!/usr/bin/env python3
"""Run only the shape_identification custom task."""
if __package__:
    from .task import TASK
else:
    from task import TASK

if __name__ == '__main__':
    TASK.cli()
