# Tips and tricks

Small, practical stuff that's saved time on this project. Running list, linked from the main README.

- **`pipreqs`, for `pyproject.toml` dependencies.** Scans actual `import` statements in the codebase and prints a minimal dependency list, instead of hand-copying whatever happens to be sitting in the environment. `pipreqs . --print`, then hand-check and paste into `pyproject.toml`.
- **`pip freeze > requirements.txt`, for a fully pinned, exact snapshot.** Different job than `pyproject.toml`: this one's the frozen full picture (every transitive dependency, exact versions), the other is the short curated list of what the code directly imports.
- **`conda env export --from-history > environment.yml`, if using conda.** Only records what was explicitly asked for on install, not the full resolved tree with build strings. Keeps the file short and readable instead of machine-specific and broken elsewhere.
- **Cross-check Polars against pandas on a small slice**, for anything non-trivial: rolling windows, shifts, joins. Pandas' rolling API is mature and well understood, so it's a fast way to catch a Polars bug (wrong shift direction, missing `.over()`, wrong sort assumption) before it goes further downstream.
- **Sort explicitly, every time, right after loading data.** Never trust that a parquet file is sorted just because some earlier script sorted it before writing it. Cheap insurance against a whole category of subtle bugs.
- **Test on a tiny synthetic frame with hand-computed expected values**, before trusting a feature on the full dataset. Slower to write than "just run it," much faster than debugging a wrong number three pipeline stages downstream.