# Contributing to Arqyon MarketCast

Contributions are by invitation. The repository owner approves who can
access the private repository and who can push changes. Keep source code,
patches, and project discussions within approved channels.

## Submit a change

1. Create a branch from the default branch. Use a short name such as
   `fix/trendlens-volume`.
2. Keep the change focused. For TrendLens, preserve the CPU-only workflow
   and chronological evaluation.
3. For code changes, run the relevant checks. To smoke-test TrendLens from
   its project folder on Windows, run:

   ```powershell
   .\.venv\Scripts\python.exe predictor.py --no-show
   ```

4. Push your branch and open a pull request in the private repository.
   Explain what changed and how you verified it. Ask the repository owner
   or a designated maintainer to review it before merging.

Keep credentials, local virtual environments, and generated model files
out of commits.
