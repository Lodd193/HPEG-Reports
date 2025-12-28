# Executive Intelligence Report - Design Guide
**HPEG Reporting System - Insights Slide**
**Version:** 1.4 - Executive-Ready Design

---

## **🎯 WHAT SENIOR LEADERSHIP WILL SEE**

The **Intelligence Report: Key Trends & Patterns** slide (Slide 6) provides executives with:

1. **Executive Summary Bar** (Blue banner at top)
   - Total significant trends identified
   - High priority count
   - Breakdown of increases vs decreases

2. **Areas of Increase** (Red section)
   - Top 3 most significant complaint increases
   - Each shown as professional card with:
     - CDG name + Priority badge
     - Trend arrow + absolute/percentage change
     - Main complaint themes
     - **Actionable recommendation**

3. **Areas of Decrease** (Green section)
   - Top 2 most significant complaint decreases
   - Positive trends worth celebrating/replicating

---

## **📊 INSIGHT CARD ANATOMY**

Each insight card contains **3 critical lines**:

### **Line 1: CDG + Priority + Change**
```
CDG8 [HIGH PRIORITY] ↑ 12 cases (85%)
```
- **CDG name** in NHS dark blue, bold
- **Priority badge** in matching color [HIGH PRIORITY / MEDIUM / MONITOR / SIGNIFICANT / POSITIVE]
- **Trend arrow** ↑ increase / ↓ decrease
- **Absolute change** (12 cases) and **percentage** (85%)

### **Line 2: Main Themes**
```
Main themes: Communication with patients, values and behaviours of staff, clinical treatment
```
- Top 3 subjects driving the CDG trend
- Provides context for "why" complaints are changing

### **Line 3: Actionable Recommendation**
```
→ Recommend immediate deep-dive review with clinical leads to identify root causes
```
- **Arrow indicator** → for visual clarity
- **Italic blue text** for distinction
- **Specific action** based on priority level

---

## **🎨 VISUAL DESIGN FEATURES**

### **Color Coding by Impact:**

**Increases (Red spectrum):**
- **High Priority** (≥10 case change): Dark red border, light red background
- **Medium** (5-9 cases): Orange border, light orange background
- **Monitor** (3-4 cases): Yellow border, light yellow background

**Decreases (Green spectrum):**
- **Significant** (≥10 case reduction): Green border, light green background
- **Positive** (<10 cases): Grey border, light grey background

### **NHS Branding:**
- All colors from official NHS palette
- Rounded rectangle cards for modern look
- 2.5pt colored borders for emphasis
- Subtle tinted backgrounds for differentiation

### **Typography Hierarchy:**
- **12pt bold** - Executive summary title
- **11pt bold** - Section headers (▲ AREAS OF INCREASE / ▼ AREAS OF DECREASE)
- **11pt bold** - CDG names
- **10pt bold** - Changes and statistics
- **9pt regular** - Themes
- **9pt italic** - Recommendations

---

## **🔍 INTELLIGENT ALGORITHM**

### **Significance Thresholds:**

A trend is flagged as significant if it meets **ANY** of these criteria:

1. **High Volume CDG** (>10 cases/month) with **>15% change**
   - Example: CDG2 went from 20 to 25 cases (25% increase) → Flagged

2. **Medium Volume CDG** (5-10 cases/month) with **>25% change**
   - Example: CDG8 went from 8 to 12 cases (50% increase) → Flagged

3. **Large Absolute Change** (**≥5 cases** regardless of percentage)
   - Example: CDG1 went from 50 to 56 cases (12% increase, but +6 cases) → Flagged

4. **Dramatic Small Volume** (<5 cases/month) with **>50% change AND ≥3 cases**
   - Example: CDG12 went from 2 to 5 cases (150% increase, +3 cases) → Flagged

**Why these thresholds?**
- Filters out noise (minor fluctuations)
- Captures meaningful trends at all volume levels
- Balances percentage changes with absolute impact
- Ensures exec team sees only actionable insights

---

## **💬 EXAMPLE INSIGHTS**

### **High Priority Increase:**
```
╔══════════════════════════════════════════════════════════════════╗
║ CDG2 [HIGH PRIORITY] ↑ 12 cases (85%)                          ║
║ Main themes: Waiting times, admission/discharge, communication  ║
║ → Recommend immediate deep-dive review with clinical leads to   ║
║   identify root causes                                          ║
╚══════════════════════════════════════════════════════════════════╝
```
**Border:** Dark red | **Background:** Light red tint

### **Medium Priority Increase:**
```
╔══════════════════════════════════════════════════════════════════╗
║ CDG8 [MEDIUM] ↑ 7 cases (45%)                                  ║
║ Main themes: Clinical treatment, consent, medication           ║
║ → Monitor closely and schedule review if trend continues next  ║
║   month                                                         ║
╚══════════════════════════════════════════════════════════════════╝
```
**Border:** Orange | **Background:** Light orange tint

### **Significant Decrease:**
```
╔══════════════════════════════════════════════════════════════════╗
║ CDG5 [SIGNIFICANT] ↓ 11 cases (55%)                            ║
║ Main themes: Patient property, lost items, security            ║
║ → Significant improvement - recommend documenting interventions║
║   for replication                                               ║
╚══════════════════════════════════════════════════════════════════╝
```
**Border:** Green | **Background:** Light green tint

---

## **📋 ACTIONABLE RECOMMENDATIONS**

The system automatically generates context-aware recommendations:

### **For Increases:**

| Priority | Recommendation |
|----------|---------------|
| **High** | "Recommend immediate deep-dive review with clinical leads to identify root causes" |
| **Medium** | "Monitor closely and schedule review if trend continues next month" |
| **Low** | "Continue monitoring - early trend indication" |

### **For Decreases:**

| Priority | Recommendation |
|----------|---------------|
| **High** | "Significant improvement - recommend documenting interventions for replication" |
| **Medium/Low** | "Positive trend - continue current approach" |

**Why this matters:**
- Executives know **exactly what to do** with each insight
- No ambiguity - clear next steps provided
- High-priority items get urgent attention
- Positive trends get recognition/replication

---

## **🎯 LAYOUT OPTIMIZATION**

### **Slide Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│ [Blue Title Bar] Intelligence Report: Key Trends & Patterns│
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ EXECUTIVE SUMMARY: 5 significant trends identified     │ │
│ │ (2 high priority) • 3 increases, 2 decreases           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ▲ AREAS OF INCREASE                                        │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ CDG2 [HIGH] ↑ 12 cases (85%)                         │   │
│ │ Themes: ...                                           │   │
│ │ → Action: ...                                         │   │
│ └───────────────────────────────────────────────────────┘   │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ CDG8 [MEDIUM] ↑ 7 cases (45%)                        │   │
│ └───────────────────────────────────────────────────────┘   │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ CDG3 [MONITOR] ↑ 4 cases (25%)                       │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                             │
│ ▼ AREAS OF DECREASE                                        │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ CDG5 [SIGNIFICANT] ↓ 11 cases (55%)                  │   │
│ └───────────────────────────────────────────────────────┘   │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ CDG1 [POSITIVE] ↓ 5 cases (15%)                      │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Spacing:**
- 0.85" summary banner
- 0.45" section headers
- 0.85" per insight card
- 0.95" spacing between cards
- 0.15" spacing between sections

---

## **✅ QUALITY CHECKLIST**

Before presenting to senior leadership, verify:

- [ ] Executive summary shows correct count of trends
- [ ] High priority items are genuinely significant (≥10 case change)
- [ ] All CDG names are spelled correctly
- [ ] Subject themes make logical sense
- [ ] Priority badges match severity (red for high increases, green for big decreases)
- [ ] Trend arrows point correct direction (↑ increase, ↓ decrease)
- [ ] Numbers are accurate (cross-check with Slide 4 CDG breakdown)
- [ ] Recommendations are actionable and appropriate
- [ ] No spelling/grammar errors in narratives
- [ ] Cards don't overflow slide boundaries

---

## **📈 EXPECTED OUTCOMES**

**For Executive Teams:**
✅ **Immediate clarity** - Know what's changing at a glance
✅ **Prioritized focus** - High-priority items stand out visually
✅ **Actionable** - Clear next steps for each trend
✅ **Contextual** - Understand themes driving changes
✅ **Professional** - Presentation-ready for board meetings

**For Report Authors:**
✅ **Time-saving** - No manual trend analysis needed
✅ **Consistent** - Same methodology applied across all HPEGs
✅ **Defensible** - Algorithm-driven, not subjective
✅ **Comprehensive** - All significant trends captured

---

## **🔧 TECHNICAL NOTES**

### **Data Processing:**
- File: `hpeg_data_processor.py`
- Function: `generate_narrative_insights()` (lines 791-936)
- Runs automatically during metrics calculation
- Adds ~3 seconds processing time per HPEG

### **Slide Generation:**
- File: `hpeg_report_generator.py`
- Functions: `create_slide_narrative_insights()` + `_create_insight_card()` (lines 791-1020)
- Uses python-pptx library for PowerPoint creation
- All NHS colors from official palette

### **Algorithm Performance:**
- Typically identifies 3-8 insights per HPEG
- Top 3 increases shown (most impactful)
- Top 2 decreases shown (celebrating wins)
- Maximum 5 cards per slide (prevents overcrowding)

---

## **🚀 FUTURE ENHANCEMENTS (Potential)**

1. **Trend Sparklines** - Mini graphs showing 6-month trend history
2. **Benchmark Comparisons** - Compare to trust average
3. **Recurring Patterns** - Flag CDGs that increase consistently
4. **Specialty Drill-Down** - Click CDG to see specialty breakdown
5. **Export to Word** - Auto-generate narrative summary document

---

## **📞 SUPPORT**

If insights seem incorrect:
1. Check data quality in source CSV (missing CDG values?)
2. Verify date range entered correctly (November 2025 not 2024)
3. Review Slide 4 CDG breakdown to confirm numbers
4. Check console output for algorithm decisions

---

**End of Executive Insights Guide**
**Designed for Maximum Impact in Senior Leadership Presentations**
