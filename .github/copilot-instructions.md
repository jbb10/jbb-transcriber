## Commit Format (REQUIRED)
Use Conventional Commits for automated versioning:
- `fix:` → patch (0.0.X)
- `feat:` → minor (0.X.0)  
- `feat!:` / `fix!:` / `BREAKING CHANGE:` → major (X.0.0)
- `chore:`, `docs:`, `test:`, `refactor:`, `style:` → no bump

## Code Standards
- Python 3.10+, type hints required
- Run `make lint` and `make test` before committing
- Follow TDD: failing test → minimal code → passing test
