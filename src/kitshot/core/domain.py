"""Shared state vocabulary for planning, persistence and job execution."""

from typing import Literal, get_args

type GroupName = Literal["a", "b", "c"]
type JobKind = Literal["initial", "kit", "age", "upgrade", "reroll", "redraw"]
type JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
type FinishedStatus = Literal["done", "failed"]

JOB_KINDS = get_args(JobKind.__value__)
JOB_STATUSES = get_args(JobStatus.__value__)
ACTIVE_JOB_STATUSES = ("queued", "running")
