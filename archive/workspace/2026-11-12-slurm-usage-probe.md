# 2026-11-12 — Slurm usage probe with zero active credits

The user authorized bounded Slurm test jobs regardless of the active-credit
balance. `AGENTS.md` now treats the `hyakusage` balance as context rather than a
submission gate; scheduler acceptance and actual accounting remain authoritative.

Before submission, `hyakusage -a video` showed $0 active credits and 20 jobs
for `xuebiz` at $1.19 in the current cycle. `sinfo -s` and `squeue --me` were
checked. Job **458819** was submitted with account `video`, partition
`gpu-h200-mig`, QoS `debug`, one `h200_1g.18gb` GPU, one CPU, 2 GB RAM, and a
two-minute wall limit. Slurm accepted it with a $0.00 rounded estimate. The
job checked GPU visibility and slept five seconds; no model or simulator ran.

Slurm reports `COMPLETED`, exit `0:0`, node `g024`, and nine seconds elapsed.
Its allocated TRES included one `h200_1g.18gb` GPU. The post-run usage query
showed 21 user jobs and the same displayed $1.19 user cost; cent rounding
prevents assigning an exact price to this short job. The account-wide total
changed while other users were active and is not this job's cost. `nvidia-smi`
listed an NVIDIA H200; this probe does not establish GPU compute performance or
MIG isolation.

The submitted script, `run_metadata.json`, Slurm accounting, post-run usage
snapshot, and output are retained under
`05_Experiment/results/slurm-usage-probe-458819/`. The Slurm output also
remains at `05_Experiment/logs/slurm-usage-probe-458819.out`. These runtime
artifacts are outside the source Git snapshot.
