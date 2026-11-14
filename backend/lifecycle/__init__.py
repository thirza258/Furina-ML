"""The experiment-and-model-lifecycle app for Furina ML.

This app owns the metadata the PRD calls the "lifecycle": dataset versions
and lineage, experiments, runs, the model registry, deployments, monitoring
records and the job queue. The machine learning itself stays in `core.ml`,
which is Django-free; the rule the whole split rests on is that Django
orchestrates the ML, it never *is* the ML engine.
"""
