# Virtual Environment Setup & Usage

## ⚠️ CRITICAL: Always Use .venv

This project **requires** a Python virtual environment (`.venv`). **Never** use system Python or `python3` commands directly.

## Initial Setup

### 1. Create Virtual Environment (First Time Only)
```bash
# Navigate to project root
cd /home/kaue/landslide_susc_modelling

# Create virtual environment
python3 -m venv .venv
```

### 2. Activate Virtual Environment
```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

When activated, your prompt will show `(.venv)` prefix.

### 3. Install Dependencies
```bash
# With venv activated:
pip install -r requirements.txt
```

## Daily Usage

### Every Terminal Session
```bash
# ALWAYS activate before running any commands!
source .venv/bin/activate
```

### Run Pipeline
```bash
# With venv activated:
python -m src.main_pipeline
```

### Run Evaluation
```bash
# With venv activated:
python -m src.evaluate --susceptibility outputs/test_susceptibility.tif \
                        --ground_truth /path/to/Ground_truth_train.tif
```

### Deactivate When Done
```bash
deactivate
```

## For Automated Tools / Agents

**NEVER** use `python` or `python3` directly. Instead:

### Option 1: Use .venv/bin/python Directly
```bash
# Direct invocation (no activation needed)
.venv/bin/python -m src.main_pipeline
.venv/bin/python -m src.evaluate --susceptibility outputs/test_susceptibility.tif
.venv/bin/pip install some-package
```

### Option 2: Activate Then Run
```bash
# Activate first
source .venv/bin/activate

# Then run normally
python -m src.main_pipeline
```

## Troubleshooting

### "Module not found" errors
**Cause**: Virtual environment not activated or dependencies not installed

**Fix**:
```bash
# Activate venv
source .venv/bin/activate

# Install/reinstall dependencies
pip install -r requirements.txt
```

### "Command not found: python"
**Cause**: Virtual environment not activated

**Fix**:
```bash
# Use direct path
.venv/bin/python -m src.main_pipeline

# OR activate venv
source .venv/bin/activate
python -m src.main_pipeline
```

### Dependencies out of date
```bash
# With venv activated:
pip install --upgrade -r requirements.txt
```

## Verification

### Check if venv is activated:
```bash
which python
# Should show: /home/kaue/landslide_susc_modelling/.venv/bin/python

echo $VIRTUAL_ENV
# Should show: /home/kaue/landslide_susc_modelling/.venv
```

### Check installed packages:
```bash
# With venv activated:
pip list
pip show rasterio  # Check specific package
```

### Test imports:
```bash
# With venv activated:
python -c "import rasterio, scipy, sklearn, torch; print('✅ All dependencies OK')"
```

## Why .venv?

1. **Isolation**: Project dependencies don't interfere with system Python
2. **Reproducibility**: Everyone uses same package versions
3. **Safety**: Can't accidentally break system Python
4. **Portability**: Easy to recreate environment on different machines

## Adding New Dependencies

```bash
# With venv activated:
pip install new-package-name

# Update requirements.txt
pip freeze > requirements.txt
```

## Recreating Environment

If `.venv` becomes corrupted:

```bash
# Remove old venv
rm -rf .venv

# Create fresh venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Reinstall everything
pip install -r requirements.txt
```

---

**Remember**: `source .venv/bin/activate` is the first command of every session! 🎯
