# Troubleshooting Guide - Topic Intelligence System
**Date:** 28 December 2025

---

## 🔧 QUICK DIAGNOSTIC

**Run this first to identify the issue:**
```bash
cd C:\Users\lod19\HPEGs

# Test 1: Check syntax
python -m py_compile hpeg_data_processor.py
python -m py_compile hpeg_report_generator.py

# Test 2: Check pickle file exists
ls -lh processed_data.pkl

# Test 3: Run data processor
python hpeg_data_processor.py
```

---

## 🐛 COMMON ERRORS & SOLUTIONS

### Error 1: TypeError: string indices must be integers, not 'str'

**Full Error:**
```
TypeError: string indices must be integers, not 'str'
  File "hpeg_data_processor.py", line 1478, in main
    critical_count = sum(1 for p in priorities if p['priority_level'] == 'CRITICAL')
```

**Cause:** `calculate_topic_priorities()` returning wrong data structure

**Solution:** ✅ ALREADY FIXED (line 1055)
- If you see this, you're using OLD code
- Check that line 1055 says `return priorities` NOT `return {...}`

**How to Verify Fix:**
```bash
grep -n "return priorities" hpeg_data_processor.py
# Should show line 1055: return priorities
```

---

### Error 2: RuntimeWarning: invalid value encountered in divide

**Full Error:**
```
RuntimeWarning: invalid value encountered in divide
  topic_dist = topic_dist / topic_dist.sum()
```

**Cause:** Division by zero when topic distribution sums to 0

**Solution:** ✅ ALREADY FIXED (lines 541-543)
- Added check: `if total > 0: topic_dist = topic_dist / total`

**How to Verify Fix:**
```bash
grep -A 2 "total = topic_dist.sum()" hpeg_data_processor.py
# Should show:
#   total = topic_dist.sum()
#   if total > 0:
#       topic_dist = topic_dist / total
```

---

### Error 3: KeyError: 'topic_analysis'

**Full Error:**
```
KeyError: 'topic_analysis'
  File "hpeg_report_generator.py", line 2318, in generate_hpeg_report
    topic_priorities = data.get('topic_analysis', {}).get(hpeg_name, [])
```

**Cause:** Old pickle file doesn't have topic_analysis key

**Solution:**
1. Delete old pickle file:
   ```bash
   rm processed_data.pkl
   ```

2. Re-run data processor:
   ```bash
   python hpeg_data_processor.py
   ```

3. Try report generation again:
   ```bash
   python hpeg_report_generator.py
   ```

**Prevention:** Always re-run data processor after code changes

---

### Error 4: IndexError: list index out of range

**Full Error:**
```
IndexError: list index out of range
  (various locations involving priorities[0] or similar)
```

**Cause:** Empty priorities list (HPEG has < 10 complaints)

**Solution:** This is EXPECTED behavior
- HPEGs with < 10 complaints skip topic analysis
- Slide 7 will show "Insufficient data" message instead
- No action needed

**To Check:**
```bash
# Look for lines like:
# "⚠ SH Exec Team: Only 26 complaints - skipping topic analysis"
```

---

### Error 5: All Topics Have Same Label

**Observation:**
```
Topic 1: Access / Wrong / Write
Topic 2: Access / Wrong / Write
...
```

**Cause:** One or more of:
- Stopwords removing too many meaningful terms
- Text preprocessing too aggressive
- Data has limited vocabulary diversity
- Too many topics for available text

**Solution Options:**

**Option A: Reduce Number of Topics**
```python
# In hpeg_data_processor.py, line ~465
# Change from:
n_topics = 10
# To:
n_topics = 6
```

**Option B: Review Stopwords**
```python
# In hpeg_data_processor.py, search for DOMAIN_STOPWORDS
# Remove some words that might be meaningful in your context
```

**Option C: Add Bigrams**
```python
# In preprocess_for_topics(), after vectorizer creation:
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Add this line for bigrams
    stop_words=custom_stopwords
)
```

**Status:** NON-CRITICAL - System still works, just less readable

---

### Error 6: FileNotFoundError: processed_data.pkl

**Full Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'processed_data.pkl'
```

**Cause:** Data processor hasn't been run yet

**Solution:**
```bash
python hpeg_data_processor.py
```

**Prevention:** Always run data processor before report generator

---

### Error 7: Slide 7 is Blank/Missing

**Observation:** Report opens but Slide 7 has no content or is skipped

**Diagnostic Steps:**

1. **Check console output from data processor:**
   ```
   Did you see:
   "✓ BHH Exec Team: X CRITICAL, Y MONITOR priorities identified"
   ```
   - YES → Data exists, check step 2
   - NO → HPEG has < 10 complaints (expected behavior)

2. **Check pickle file has data:**
   ```python
   import pickle
   with open('processed_data.pkl', 'rb') as f:
       data = pickle.load(f)
   print('topic_analysis' in data)  # Should be True
   print(data['topic_analysis'].keys())  # Should show HPEG names
   ```

3. **Check report generator ran without errors:**
   - Look for: "Generating report for [HPEG]..."
   - Should NOT see errors about missing keys

**Common Causes:**
- HPEG has insufficient data (< 10 complaints) → Shows "Insufficient data" message
- Old pickle file → Delete and regenerate
- Code changes not saved → Verify files saved

---

## 🔍 DEBUGGING WORKFLOW

### Step-by-Step Diagnostic Process

**1. Verify Code is Up-to-Date**
```bash
# Check these specific lines:
grep -n "return priorities" hpeg_data_processor.py | grep 1055
# Should show: 1055:    return priorities

grep -n "if total > 0:" hpeg_data_processor.py | grep 542
# Should show: 542:    if total > 0:
```

**2. Clean Start**
```bash
# Remove old outputs
rm -f processed_data.pkl
rm -f outputs/*.pptx

# Run fresh
python hpeg_data_processor.py 2>&1 | tee data_processor_log.txt
python hpeg_report_generator.py 2>&1 | tee report_generator_log.txt
```

**3. Check Logs**
```bash
# Data processor log should show:
grep "STEP 3B" data_processor_log.txt
grep "priorities identified" data_processor_log.txt

# Report generator log should show:
grep "Slide 7" report_generator_log.txt
grep "Topic Intelligence" report_generator_log.txt
```

**4. Inspect Pickle Contents**
```python
import pickle
import json

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Check structure
print("Keys:", data.keys())
print("\nTopic Analysis HPEGs:", data.get('topic_analysis', {}).keys())

# Check one HPEG
bhh = data.get('topic_analysis', {}).get('BHH Exec Team', [])
print(f"\nBHH has {len(bhh)} priority topics")
if len(bhh) > 0:
    print("\nFirst priority topic:")
    print(json.dumps(bhh[0], indent=2, default=str))
```

**5. Visual Check**
- Open generated .pptx file
- Count slides (should be 19)
- Check Slide 7 title bar
- Look for colored priority boxes

---

## 🔬 ADVANCED DEBUGGING

### Enable Verbose Topic Analysis
```python
# In hpeg_data_processor.py, add after line 1466:
print(f"\n  DEBUG: Topic performance for {hpeg}:")
for tp in topic_perf[:3]:  # Show first 3
    print(f"    Topic {tp['topic_id']}: {tp['topic_label']}")
    print(f"      Prevalence: {tp['prevalence_pct']:.1f}%")
    print(f"      Resolution: {tp['median_resolution_days']:.1f} days")
```

### Check Chart Generation
```python
# In hpeg_report_generator.py, add after line 2315:
print(f"  DEBUG: topic_priorities type: {type(topic_priorities)}")
print(f"  DEBUG: topic_priorities length: {len(topic_priorities)}")
if len(topic_priorities) > 0:
    print(f"  DEBUG: First priority: {topic_priorities[0].get('topic_label')}")
```

### Validate Priority Scores
```python
# After calculate_topic_priorities() returns:
for p in priorities[:5]:
    print(f"{p['topic_label']}: Score={p['priority_score']:.2f}, Level={p['priority_level']}")
```

---

## 📊 HEALTH CHECK SCRIPT

Save this as `health_check.py` for quick diagnostics:

```python
#!/usr/bin/env python3
"""Quick health check for Topic Intelligence system."""

import pickle
import sys
from pathlib import Path

def health_check():
    print("="*60)
    print("TOPIC INTELLIGENCE HEALTH CHECK")
    print("="*60)

    # Check 1: Pickle file exists
    print("\n1. Checking processed_data.pkl...")
    if Path('processed_data.pkl').exists():
        print("   ✓ File exists")
        size_mb = Path('processed_data.pkl').stat().st_size / (1024*1024)
        print(f"   ✓ Size: {size_mb:.1f} MB")
    else:
        print("   ✗ File missing - run hpeg_data_processor.py")
        return False

    # Check 2: Load pickle
    print("\n2. Loading pickle file...")
    try:
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("   ✓ Loaded successfully")
    except Exception as e:
        print(f"   ✗ Load failed: {e}")
        return False

    # Check 3: Topic analysis key exists
    print("\n3. Checking topic_analysis key...")
    if 'topic_analysis' in data:
        print("   ✓ Key exists")
        hpegs = list(data['topic_analysis'].keys())
        print(f"   ✓ HPEGs with data: {len(hpegs)}")
        for hpeg in hpegs:
            count = len(data['topic_analysis'][hpeg])
            print(f"      - {hpeg}: {count} priority topics")
    else:
        print("   ✗ Key missing - re-run data processor")
        return False

    # Check 4: Verify structure
    print("\n4. Checking data structure...")
    for hpeg, priorities in data['topic_analysis'].items():
        if len(priorities) > 0:
            first = priorities[0]
            required_keys = ['topic_id', 'topic_label', 'priority_score',
                           'priority_level', 'priority_color', 'recommendation']
            missing = [k for k in required_keys if k not in first]
            if missing:
                print(f"   ✗ {hpeg} missing keys: {missing}")
                return False
    print("   ✓ All required keys present")

    # Check 5: Output folder
    print("\n5. Checking outputs folder...")
    if Path('outputs').exists():
        pptx_files = list(Path('outputs').glob('*.pptx'))
        print(f"   ✓ Found {len(pptx_files)} .pptx files")
    else:
        print("   ⚠ No outputs folder - reports not yet generated")

    print("\n" + "="*60)
    print("HEALTH CHECK PASSED ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    success = health_check()
    sys.exit(0 if success else 1)
```

**Run it:**
```bash
python health_check.py
```

---

## 🆘 WHEN ALL ELSE FAILS

### Nuclear Option: Full Reset
```bash
# Backup current work
cp hpeg_data_processor.py hpeg_data_processor.BACKUP.py
cp hpeg_report_generator.py hpeg_report_generator.BACKUP.py

# Clean everything
rm -f processed_data.pkl
rm -f outputs/*.pptx
rm -f *.log

# Start fresh
python hpeg_data_processor.py
python hpeg_report_generator.py
```

### Get Help
If still stuck, provide this information:

1. **Error message** (full traceback)
2. **Output of health_check.py**
3. **Last 50 lines of data processor output:**
   ```bash
   python hpeg_data_processor.py 2>&1 | tail -50
   ```
4. **Topic analysis sample:**
   ```python
   import pickle
   with open('processed_data.pkl', 'rb') as f:
       data = pickle.load(f)
   print(data.get('topic_analysis', {}).get('BHH Exec Team', [])[:2])
   ```

---

## 📚 REFERENCE LINKS

- **Full Session Log:** `SESSION_LOG_2025-12-28.md`
- **Quick Start:** `QUICK_START_TOMORROW.md`
- **Implementation Progress:** `IMPLEMENTATION_PROGRESS.md`

---

**Last Updated:** 28 December 2025
**Status:** Ready for testing with bug fixes applied
