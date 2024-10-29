# Contributing to residue-estimator

Thank you for contributing! This guide outlines expectations for an effective collaborative workflow using Git, with conventions for commit messages and branches.

> **Note** Given the small size of our team, these conventions are meant to serve as a flexible guide. Please refer to them for larger development efforts, such as core features or major updates. However, for smaller, trivial changes, you may skip certain steps, like the branching strategy and pull requests, when they might otherwise slow down progress. The goal is to prioritize productivity without sacrificing team standards.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/residue-estimator.git
   cd residue-estimator
   ```

2. **Install Project Dependencies**
   Make sure to follow the README for setting up dependencies and Docker. 

## Branching Strategy

We use a branch-per-feature approach. Follow this convention:

- **main**: The stable branch; it should always be deployable.
- **feature/your-feature-name**: For new features or enhancements.
- **bugfix/issue-description**: For bug fixes.
- **hotfix/urgent-fix**: For critical fixes on production code.

### Example
If adding a data parser feature, your branch could be named `feature/data-parser`.

## Commit Message Guidelines

When committing changes, please follow these conventions to ensure clarity and consistency:

- Use the present tense: “Add feature” instead of “Added feature.”
- Keep the subject line to 50 characters or less.
- Use the body to explain the "why" behind the changes if necessary, wrapping it at 72 characters.
- Reference any related issues using `#issue-number`.

More information can be found in the *Additional Resources* section.

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)