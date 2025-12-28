# Tests Directory

Unit tests for the dino_finder package.

## Purpose

Tests ensure your code works correctly and catches bugs early.

## Running Tests

```bash
# Install pytest if not already
pip install pytest

# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset.py

# Run with verbose output
pytest -v
```

## Test Structure

Mirror the `src/` structure:
```
tests/
├── test_models.py      # Tests for src/dino_finder/models/
├── test_dataset.py     # Tests for src/dino_finder/data/
└── test_utils.py       # Tests for src/dino_finder/utils/
```

## Example Test

```python
# tests/test_dataset.py
import pytest
from dino_finder.data import DinoDataset

def test_dataset_length():
    dataset = DinoDataset("data/train")
    assert len(dataset) > 0
```

## Good Practices

- Write tests for critical functions
- Test edge cases (empty inputs, invalid data, etc.)
- Keep tests simple and focused
- Run tests before committing code
