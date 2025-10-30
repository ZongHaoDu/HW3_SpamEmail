## Why
The repository currently lacks a filled `openspec/project.md` describing project goals, tech stack, and conventions. A complete project context helps contributors and automated agents (like OpenSpec-enabled assistants) quickly understand assumptions, constraints, and the intended workflow.

## What Changes
- Add a populated `openspec/project.md` with purpose, tech stack, conventions, testing strategy, git workflow, domain context, constraints, and external dependencies.
- State assumptions made during authoring so reviewers can correct any inaccuracies.

## Impact
- Affected files:
  - `openspec/project.md` (new content)
  - `openspec/changes/add-project-context/proposal.md` (this file)
  - `openspec/changes/add-project-context/tasks.md` (this file)
- Affected parties: contributors, TAs, and automated assistants that rely on OpenSpec metadata.

**BREAKING**: None. This is documentation-only.

## Validation
- Reviewer should confirm the tech stack and conventions match the project's actual implementation.
- Run `openspec validate add-project-context --strict` (if using OpenSpec CLI) once deltas exist. This proposal is documentation-only and doesn't include spec deltas.

## Notes
- If the project uses a different language/tooling (e.g., Node.js), the project.md will be updated to reflect that.