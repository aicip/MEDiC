# Contributing to MEDiC

Thank you for your interest in contributing!

## Reporting Issues

- **Bug reports**: Include the full error traceback, your PyTorch/CUDA version, and the command you ran.
- **Feature requests**: Describe the use case and expected behavior.

## Pull Requests

1. Fork the repo and create a feature branch from `main`
2. Make your changes and ensure tests pass: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py -v`
3. Submit a PR with a clear description of what changed and why

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/MEDiC.git
cd MEDiC
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py -v
```

All 25 tests should pass. Tests use tiny models and run on CPU.

## Code Style

- Follow existing patterns in the codebase
- Keep imports at the top of files (except CLIP, which is lazy-loaded for test speed)
