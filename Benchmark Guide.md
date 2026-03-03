# SCBI Benchmark Suite - User Guide

## 📊 Publication-Quality Proof of Concept

This comprehensive benchmark suite provides rigorous experimental validation of SCBI for research papers and repository documentation.

---

## 🎯 What This Benchmark Provides

### ✅ Statistical Rigor
- **Multiple runs** (5 independent trials per dataset)
- **Statistical significance testing** (paired t-tests)
- **Confidence intervals** (mean ± std)
- **Effect size reporting** (improvement percentages)

### ✅ Comprehensive Coverage
- **5 diverse datasets** (regression + classification)
- **Real-world data** (California Housing, Diabetes, Breast Cancer)
- **Synthetic data** (controlled high-dimensional tests)
- **Multiple tasks** (regression and classification)

### ✅ Publication-Ready Outputs
- **High-resolution figures** (300 DPI PNG + vector PDF)
- **LaTeX tables** (ready to paste into papers)
- **CSV data** (for further analysis)
- **JSON results** (structured data)

### ✅ Ablation Studies
- Effect of **n_samples** (number of bagging subsets)
- Effect of **sample_ratio** (subset size)
- Effect of **ridge_alpha** (regularization strength)

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy

# Download SCBI
wget https://zenodo.org/record/18576203/files/scbi.py

# Run benchmark
python benchmark_publication.py
```

### Expected Runtime

| Component | Time |
|-----------|------|
| Per dataset | ~5-10 minutes |
| Total (5 datasets) | ~30-45 minutes |
| Visualizations | ~2 minutes |
| **Total** | **~35-50 minutes** |

---

## 📁 Output Files

### Figures (High-Resolution)

```
figures/
├── fig1_training_curves.png       # Training curves with confidence intervals
├── fig1_training_curves.pdf       # Vector version for papers
├── fig2_initial_loss.png          # Initial loss comparison bar chart
├── fig2_initial_loss.pdf
├── fig3_convergence_speedup.png   # Epochs to convergence boxplot
├── fig3_convergence_speedup.pdf
├── fig4_significance.png          # Statistical significance visualization
├── fig4_significance.pdf
├── fig5_ablation.png              # Hyperparameter ablation study
└── fig5_ablation.pdf
```

### Results Data

```
results/
├── results.csv                    # Full results in CSV format
├── results_table.tex              # LaTeX table for papers
└── results.json                   # Structured JSON data
```

---

## 📊 Understanding the Figures

### Figure 1: Training Curves

**What it shows:**
- Training loss over 30 epochs
- Shaded regions = ±1 standard deviation
- Red line = SCBI initialization
- Blue line = Standard initialization

**How to interpret:**
- Lower initial loss → Better initialization
- Faster convergence → Fewer epochs needed
- Smaller shaded region → More stable across runs

**For your paper:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/fig1_training_curves.pdf}
\caption{Training curves comparing SCBI vs standard initialization across 
five datasets. Solid lines show mean loss over 5 independent runs, 
shaded regions indicate ±1 standard deviation.}
\label{fig:training_curves}
\end{figure}
```

### Figure 2: Initial Loss Comparison

**What it shows:**
- Bar chart comparing initial loss (epoch 0)
- Error bars = standard deviation
- Percentages above = improvement

**How to interpret:**
- Shorter SCBI bars = better initialization
- Large percentage = big improvement
- Small error bars = consistent performance

**Key metric:** Initial loss improvement percentage

### Figure 3: Convergence Speedup

**What it shows:**
- Boxplots showing epochs to reach target loss
- Target = 2× SCBI's initial loss
- Lower = faster convergence

**How to interpret:**
- SCBI boxes much lower = faster convergence
- Title shows speedup multiplier (e.g., "10× faster")
- Median line in box = typical performance

**For your paper:**
```
SCBI achieves X× faster convergence, reaching target loss 
in N epochs vs M epochs for standard initialization.
```

### Figure 4: Statistical Significance

**What it shows:**
- Left: p-values from paired t-tests
- Right: Effect sizes (improvement %)
- Green = significant, Orange = not significant

**How to interpret:**
- p-value < 0.05 = statistically significant
- Larger improvement % = bigger effect
- All green bars = robust across datasets

**For your paper:**
```
SCBI achieves statistically significant improvements across 
all datasets (p < 0.05), with effect sizes ranging from X% to Y%.
```

### Figure 5: Ablation Study

**What it shows:**
- Three subplots showing hyperparameter sensitivity
- Orange dots = initial loss
- Blue squares = final loss
- Lower = better

**How to interpret:**
- Flat lines = robust to parameter
- U-shaped curves = optimal value in middle
- Steep changes = sensitive parameter

**Recommendations from ablation:**
- `n_samples=10`: Good balance (plateau visible)
- `sample_ratio=0.5`: Optimal tradeoff
- `ridge_alpha`: Auto-tuning recommended (varies by dataset)

---

## 📈 Results Table (LaTeX)

The benchmark generates a publication-ready LaTeX table:

```latex
\begin{table}[htbp]
\centering
\caption{SCBI Performance Across Datasets}
\label{tab:scbi_results}
\begin{tabular}{lcccc}
\toprule
Dataset & \multicolumn{2}{c}{Initial Loss} & Improvement & $p$-value \\
        & Standard & SCBI & (\%) & \\
\midrule
California Housing & 26.45±2.1 & 3.21±0.4 & 87.9* & 0.0001 \\
... (more rows)
\bottomrule
\end{tabular}
\end{table}
```

**Usage in paper:**
1. Copy from `results/results_table.tex`
2. Paste into your LaTeX document
3. Requires `\usepackage{booktabs}`

---

## 📊 CSV Data Format

`results/results.csv` contains:

| Column | Description |
|--------|-------------|
| Dataset | Dataset name |
| Task | regression or classification |
| Standard_Initial_Mean | Mean initial loss (standard) |
| Standard_Initial_Std | Std dev initial loss (standard) |
| SCBI_Initial_Mean | Mean initial loss (SCBI) |
| SCBI_Initial_Std | Std dev initial loss (SCBI) |
| Improvement_Pct | Percentage improvement |
| P_Value | Statistical significance |
| Significant | Boolean (p < 0.05) |
| Init_Time_Mean | SCBI initialization time |

**Use for:**
- Further statistical analysis in R/Python
- Creating custom visualizations
- Meta-analysis across studies

---

## 🎓 How to Use in Your Research Paper

### Abstract

```
We propose SCBI, a novel initialization method that achieves X% 
improvement in initial loss and Y× faster convergence compared to 
standard initialization across Z diverse datasets (p < 0.05).
```

### Results Section

```latex
\section{Results}

We evaluated SCBI on five datasets spanning regression and 
classification tasks. Figure~\ref{fig:training_curves} shows 
training curves across all datasets. SCBI consistently achieves 
lower initial loss (Table~\ref{tab:scbi_results}), with improvements 
ranging from X\% to Y\% (all $p < 0.05$).

Convergence analysis (Figure~\ref{fig:convergence}) reveals that 
SCBI reaches target loss in M epochs vs N epochs for standard 
initialization, representing an average speedup of Z×.

Ablation studies (Figure~\ref{fig:ablation}) demonstrate robustness 
to hyperparameter choices, with performance stable across...
```

### Methods Section

```latex
\subsection{Experimental Setup}

We conducted 5 independent runs per dataset to ensure statistical 
reliability. Models were trained for 30 epochs using Adam optimizer 
(lr=0.01). SCBI initialization used 30\% of training data as proxy 
samples, with $K=10$ bagging subsets and automatic Ridge CV tuning. 
Statistical significance was assessed using paired t-tests ($\alpha=0.05$).
```

---

## 🔧 Customization

### Change Number of Runs

```python
benchmark = BenchmarkExperiment(
    n_runs=10,      # Increase for more statistical power
    n_epochs=30,
    device='cpu'
)
```

**Recommendation:**
- Research paper: 5-10 runs
- Quick test: 3 runs
- High stakes: 20+ runs

### Change Datasets

```python
datasets = [
    'california_housing',    # Your datasets
    'your_custom_dataset'    # Add custom datasets
]
```

To add custom dataset, modify `load_dataset()` method.

### Use GPU

```python
benchmark = BenchmarkExperiment(
    n_runs=5,
    n_epochs=30,
    device='cuda'    # Use GPU (10× faster)
)
```

### Adjust Epochs

```python
benchmark = BenchmarkExperiment(
    n_runs=5,
    n_epochs=50,     # More epochs for thorough analysis
    device='cpu'
)
```

---

## 📐 Statistical Notes

### Paired t-test

We use **paired t-tests** because:
- Same dataset split across methods
- Reduces variance from data sampling
- More statistical power

**Assumptions:**
- Normality: Satisfied for n_runs ≥ 5 (Central Limit Theorem)
- Independence: Each run uses different random seed

### Effect Size

**Interpretation:**
- < 5%: Small effect
- 5-20%: Medium effect
- > 20%: Large effect
- > 50%: Very large effect (SCBI typically here!)

### Confidence Intervals

Shaded regions in plots show ±1 standard deviation:
- ~68% of runs fall within this range
- Narrower = more consistent
- Wider = more variable

---

## 🎯 Reproducibility Checklist

For publication, ensure:

- [✓] **Fixed random seed** (RANDOM_SEED = 42)
- [✓] **Multiple independent runs** (≥5)
- [✓] **Statistical testing** (p-values reported)
- [✓] **Effect sizes** (improvement % reported)
- [✓] **Confidence intervals** (error bars/shading)
- [✓] **Ablation studies** (hyperparameter sensitivity)
- [✓] **Code availability** (GitHub + DOI)
- [✓] **Data availability** (datasets cited)
- [✓] **Hardware specs** (CPU/GPU documented)

---

## 📊 Example Results Interpretation

### Good Results
```
Dataset: California Housing
Standard Initial: 26.45 ± 2.1
SCBI Initial: 3.21 ± 0.4
Improvement: 87.9%
p-value: 0.0001
Significance: ✅ YES

Interpretation: SCBI provides dramatic 87.9% improvement with 
high statistical significance. Low std dev (0.4) indicates 
consistent performance across runs.
```

### Moderate Results
```
Dataset: Some Dataset
Standard Initial: 10.5 ± 1.2
SCBI Initial: 8.3 ± 1.0
Improvement: 21.0%
p-value: 0.023
Significance: ✅ YES

Interpretation: SCBI provides moderate 21% improvement, still 
statistically significant. Reasonable for datasets with strong 
non-linear components.
```

### Concerning Results
```
Dataset: Very Non-linear
Standard Initial: 5.2 ± 0.8
SCBI Initial: 5.0 ± 0.9
Improvement: 3.8%
p-value: 0.312
Significance: ❌ NO

Interpretation: Minimal improvement, not statistically significant. 
Dataset likely too non-linear for SCBI. Consider mentioning in 
limitations section.
```

---

## 🚨 Troubleshooting

### Issue: RuntimeError during SCBI init

**Cause:** Singular matrix (rare)
**Solution:** Already handled automatically with pseudo-inverse fallback
**If persists:** Increase `ridge_alpha` manually

### Issue: Poor SCBI performance

**Check:**
1. Is data standardized? (Done automatically)
2. Is dataset too small? (Need ≥100 samples)
3. Is problem too non-linear? (Check R² from sklearn Ridge)

### Issue: Figures look wrong

**Solutions:**
- Check matplotlib backend: `plt.switch_backend('Agg')`
- Increase DPI: Modify `plt.rcParams['figure.dpi'] = 300`
- Install missing fonts: `sudo apt-get install fonts-dejavu`

### Issue: Slow execution

**Solutions:**
- Use GPU: `device='cuda'`
- Reduce n_runs: `n_runs=3`
- Reduce n_epochs: `n_epochs=20`
- Use fewer datasets

---

## 📚 Citation

When using this benchmark in your research:

```bibtex
@software{ashraf2026scbi_benchmark,
  author = {Ashraf, Fares},
  title = {SCBI Comprehensive Benchmark Suite},
  year = {2026},
  doi = {10.5281/zenodo.18576203},
  url = {https://doi.org/10.5281/zenodo.18576203}
}
```

---

## 🎓 Best Practices for Paper

### Do's ✅
- Report mean ± std for all metrics
- Include p-values and significance markers (*)
- Show training curves with confidence intervals
- Discuss limitations (when SCBI doesn't help)
- Provide ablation studies
- Make code/data available

### Don'ts ❌
- Cherry-pick best runs
- Report only final loss (hide poor initial performance)
- Ignore statistical significance
- Omit error bars
- Hide negative results

---

## 🌟 Example Papers Using Similar Methodology

### Recommended Structure

1. **Introduction**: Problem statement, SCBI overview
2. **Methods**: Algorithm, implementation details
3. **Experimental Setup**: Datasets, baselines, metrics
4. **Results**: 
   - Main results (Table + Figure)
   - Training curves (Figure)
   - Ablation studies (Figure)
   - Statistical analysis (Table)
5. **Discussion**: When SCBI works/fails, why
6. **Conclusion**: Summary, future work

### Metrics to Report

**Primary:**
- Initial loss (mean ± std)
- Improvement percentage
- p-value

**Secondary:**
- Final test loss
- Epochs to convergence
- Initialization time
- Speedup multiplier

---

## 📬 Support

**Questions about the benchmark?**
- Open an issue on GitHub
- Email: farsashraf44@gmail.com
- Include: Python version, error message, dataset name

---

## ✨ Quick Command Reference

```bash
# Full benchmark (recommended)
python benchmark_publication.py

# Quick test (3 runs, 20 epochs)
# Edit script: n_runs=3, n_epochs=20

# GPU acceleration
# Edit script: device='cuda'

# View results
ls figures/          # PNG and PDF files
ls results/          # CSV, LaTeX, JSON

# Include in paper
# 1. Copy figures/*.pdf to paper/figures/
# 2. Copy results/results_table.tex to paper/
# 3. \input{results_table.tex}
```
