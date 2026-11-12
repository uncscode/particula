"""Keep the adjacent wall-loss contract suite out of recursive collection.

The top-level dynamics suite covers the shared strategy behavior. This adjacent
suite also covers charged models and stochastic array helpers and must run
explicitly by path. Separate invocations avoid same-basename import collisions;
see AGENTS.md. These are maintained native tests, not obsolete facade tests.
"""

collect_ignore_glob = ["wall_loss_strategies_test.py"]
