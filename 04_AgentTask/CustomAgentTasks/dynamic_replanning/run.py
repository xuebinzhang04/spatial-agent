#!/usr/bin/env python3
"""Run only the dynamic_replanning custom task."""
if __package__:
    from .task import TASK
else:
    from task import TASK

if __name__ == '__main__':
    TASK.cli()
