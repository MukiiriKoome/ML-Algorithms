# Lasso Regression Learning Project: Gene Expression Analysis

## Project Overview

You are a bioinformatician working on a cancer research project. You have gene expression data from 100 patients and need to predict a health outcome (disease severity score: 0-100) based on 20 different genes.

**The Challenge**: Most genes are probably irrelevant to the disease. Using regular regression would keep all 20 genes, making the model hard to interpret and potentially capturing noise. You need to use **Lasso regression** to automatically identify which genes actually matter.

---

## Dataset Description

You have 100 patients with measurements for 20 genes and a disease severity score.

**Files you'll work with:**
- `gene_expression.csv` - Gene expression data (100 rows × 20 gene columns)
- `disease_severity.csv` - Target variable (100 severity scores)

### Sample Data Structure

| Patient | Gene_1 | Gene_2 | Gene_3 | ... | Gene_20 | Severity |
|---------|--------|--------|--------|-----|---------|----------|
| 1       | 0.45   | 0.32   | 0.88   | ... | 0.12    | 45       |
| 2       | 0.52   | 0.28   | 0.91   | ... | 0.15    | 52       |
| ...     | ...    | ...    | ...    | ... | ...     | ...      |

---

## Project Timeline & Tasks

### Phase 1: Data Exploration (10 points)

1. **Load and explore the data**
   - Load gene expression and severity data
   - Check data shape, missing values, data types
   - Display basic statistics (mean, std, min, max)

2. **Visualize correlations**
   - Create a correlation matrix between genes and disease severity
   - Plot which genes have the strongest correlation with severity
   - Identify any highly correlated genes (multicollinearity)

3. **Check data distribution**
   - Plot histograms of gene expression values
   - Plot histogram of disease severity
   - Identify any outliers or skewed distributions

### Phase 2: Data Preparation (10 points)

4. **Standardize the features**
   - Standardize all 20 genes using z-score normalization
   - Center the severity scores
   - Verify standardization (mean ≈ 0, std ≈ 1)

5. **Split the data**
   - Split into training (70%, 70 patients) and test (30%, 30 patients)
   - Use `random_state=42` for reproducibility
   - Verify no data leakage

### Phase 3: Baseline Model (15 points)

6. **Fit OLS Regression** (no regularization)
   - Train on standardized training data
   - Calculate training R² and MSE
   - Calculate test R² and MSE
   - Print all 20 coefficients
   - **Question**: How many genes have non-zero coefficients? What does this tell you?

7. **Analyze OLS coefficients**
   - Which genes have the largest coefficients?
   - Are any coefficients negative?
   - Create a bar plot of OLS coefficients
   - Discuss overfitting concerns

### Phase 4: Lasso Implementation (35 points)

8. **Implement Lasso from scratch** (coordinate descent)
   
   **Algorithm to implement:**
   ```
   Function: lasso_coordinate_descent(X, y, lambda, max_iter=1000, tol=1e-4)
   
   Input: X (n×p standardized features), y (n×1 centered response), 
          lambda (regularization parameter)
   
   1. Initialize β = 0
   2. Repeat until convergence:
      For j = 1 to p:
         # Compute residual with j-th feature updated
         r = y - X·β + X_j·β_j
         
         # Compute correlation
         ρ_j = X_j^T · r / n
         
         # Apply soft-thresholding
         if ρ_j < -λ/2:
            β_j = (ρ_j + λ/2) / (1 + λ)
         elif ρ_j > λ/2:
            β_j = (ρ_j - λ/2) / (1 + λ)
         else:
            β_j = 0
      
      # Check convergence
      if ||β_new - β_old|| < tolerance: break
   
   Return β
   ```

9. **Fit Lasso for λ = 1.0**
   - Use your implemented coordinate descent algorithm
   - Record how many iterations until convergence
   - Print the coefficients
   - **Key Question**: How many genes are exactly zero? Compare to OLS!

10. **Track convergence**
    - Plot loss function vs. iterations
    - Does it converge smoothly?
    - How many iterations needed?

11. **Compute performance metrics**
    - Calculate training MSE and R²
    - Calculate test MSE and R²
    - Compare with OLS baseline

### Phase 5: Lasso Path Analysis (20 points)

12. **Fit Lasso for multiple λ values**
    - Test λ values: [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]
    - For each λ:
      - Fit Lasso model
      - Record all coefficients
      - Count number of non-zero coefficients
      - Calculate test MSE

13. **Create coefficient path plot**
    - X-axis: log(λ)
    - Y-axis: coefficient values
    - Plot a line for each gene showing how its coefficient changes
    - Mark genes that become zero
    - Title: "Lasso Coefficient Paths"

14. **Create sparsity plot**
    - X-axis: λ value
    - Y-axis: number of non-zero coefficients
    - Show how many genes are selected at each λ
    - Identify "interesting" λ values where genes drop to zero

### Phase 6: Cross-Validation (20 points)

15. **Implement 5-fold cross-validation**
    - Manually split data into 5 folds
    - For each fold:
      - Train Lasso on 4 folds
      - Evaluate on 1 fold
      - Record test MSE
    
    - Test λ values: [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]
    - Calculate mean CV error and standard deviation for each λ

16. **Plot CV results**
    - X-axis: log(λ)
    - Y-axis: Mean CV MSE with error bars (±1 std)
    - Mark optimal λ (minimum CV error)
    - Mark λ_1se (1 standard error rule: simpler model within 1 SE of best)

17. **Select optimal λ**
    - Choose λ using:
      - **Strategy 1**: Minimum CV error
      - **Strategy 2**: 1-SE rule (simpler model)
    - Justify your choice

### Phase 7: Final Model & Interpretation (20 points)

18. **Refit Lasso with optimal λ** on full training data
    - Print coefficients
    - List selected genes (non-zero coefficients)
    - Calculate training and test performance

19. **Gene Selection Results**
    - Which genes were selected?
    - Which genes were eliminated?
    - Create a summary table:
      | Gene | Coefficient | Selected? |
      |------|-------------|-----------|
      | Gene_1 | 0.234 | Yes |
      | Gene_2 | 0 | No |

20. **Biological Interpretation**
    - How many genes were selected?
    - What percentage of genes were eliminated?
    - Compare selected genes to OLS model
    - Discuss practical implications

### Phase 8: Comparison & Analysis (15 points)

21. **Compare OLS vs. Lasso**
    - Create comparison table:
      | Metric | OLS | Lasso |
      |--------|-----|-------|
      | Training MSE | ? | ? |
      | Test MSE | ? | ? |
      | Training R² | ? | ? |
      | Test R² | ? | ? |
      | Non-zero coefficients | 20 | ? |
      | Model complexity | High | Low |

22. **Analyze overfitting**
    - Is test MSE worse than training MSE for OLS?
    - Is test MSE better for Lasso?
    - Calculate overfitting ratio: (Test MSE - Train MSE) / Train MSE

23. **Answer key questions**
    - Why does Lasso produce sparse solutions while OLS doesn't?
    - What is the soft-thresholding operator and why is it important?
    - When would you prefer Lasso over OLS? When would you use Ridge instead?
    - How does the L1 penalty differ from L2 penalty geometrically?

---

## Bonus Challenges (Extra 20 points)

24. **Implement LARS algorithm**
    - Implement Least Angle Regression for Lasso
    - Compute entire solution path efficiently
    - Compare speed with coordinate descent

25. **Elastic Net comparison**
    - Implement or use Elastic Net (L1 + L2 penalty)
    - Compare with pure Lasso
    - When is Elastic Net better?

26. **Prediction on new patients**
    - Create predictions for 5 new patients with their gene expressions
    - Compare predictions from OLS vs. Lasso vs. Elastic Net
    - Discuss which model you'd trust most

27. **Sensitivity analysis**
    - Perturb the optimal λ by ±10%, ±20%
    - How sensitive are results to λ choice?
    - Which genes are consistently selected?

28. **Real-world data**
    - Find a real gene expression dataset (GEO, TCGA)
    - Apply Lasso to real biological problem
    - Write up findings

---

## Dataset Generation Code

```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 100
n_genes = 20

# Generate gene expression data (standardized)
X = np.random.randn(n_samples, n_genes)

# Create severity score based on only 5 "real" genes
true_genes = [2, 5, 7, 12, 18]  # Only these genes matter
true_coefficients = [0.8, -0.6, 0.5, 0.7, -0.4]

y = np.zeros(n_samples)
for gene, coef in zip(true_genes, true_coefficients):
    y += X[:, gene] * coef

# Add noise
y += np.random.randn(n_samples) * 0.3

# Scale to 0-100 range
y = (y - y.min()) / (y.max() - y.min()) * 100

# Create DataFrames
gene_names = [f'Gene_{i+1}' for i in range(n_genes)]
X_df = pd.DataFrame(X, columns=gene_names)

# Save to CSV
X_df.to_csv('gene_expression.csv', index=False)
y_df = pd.DataFrame({'Severity': y})
y_df.to_csv('disease_severity.csv', index=False)

print("Dataset created successfully!")
print(f"Gene expression shape: {X_df.shape}")
print(f"Severity shape: {y_df.shape}")
print(f"\nTrue relevant genes: {[gene_names[i] for i in true_genes]}")
```

---

## Deliverables Checklist

- [ ] Part 1: EDA notebook with plots
- [ ] Part 2: Data standardization and splitting
- [ ] Part 3: OLS baseline model
- [ ] Part 4: Lasso implementation (coordinate descent)
- [ ] Part 5: Coefficient path analysis with plots
- [ ] Part 6: Cross-validation results
- [ ] Part 7: Final model selection and interpretation
- [ ] Part 8: Comparison table and analysis
- [ ] Written Report (2-3 pages):
  - Problem statement
  - Methods used
  - Key findings
  - Gene selection results
  - Recommendations
- [ ] Code (well-commented)
- [ ] All plots and figures

---

## Grading Rubric

| Component | Points |
|-----------|--------|
| Data Exploration | 10 |
| Data Preparation | 10 |
| OLS Baseline | 15 |
| Lasso Implementation | 35 |
| Lasso Path Analysis | 20 |
| Cross-Validation | 20 |
| Final Model & Interpretation | 20 |
| Comparison & Analysis | 15 |
| Code Quality & Documentation | 10 |
| Written Report | 15 |
| **Total** | **170** |

Extra credit up to 20 points for bonus challenges.

---

## Learning Objectives

By completing this project, you will:

1. **Understand Lasso fundamentally**
   - Know why L1 penalty creates sparsity
   - Understand soft-thresholding operator
   - Implement coordinate descent from scratch

2. **Master feature selection**
   - Automatically identify relevant features
   - Reduce model complexity
   - Improve interpretability

3. **Apply hyperparameter tuning**
   - Perform proper cross-validation
   - Understand bias-variance tradeoff
   - Use 1-SE rule for model selection

4. **Compare regularization methods**
   - OLS vs. Ridge vs. Lasso
   - When to use each method
   - Computational considerations

5. **Practical data science skills**
   - Data preprocessing
   - Model evaluation
   - Visualization
   - Scientific communication

---

## Resources & References

**Key Concepts to Review:**
- L1 vs L2 regularization
- Soft-thresholding operator
- Coordinate descent algorithm
- Cross-validation methodology
- Sparse vs dense models

**Python Libraries:**
- `numpy` - Linear algebra
- `pandas` - Data manipulation
- `sklearn.linear_model.Lasso` - Lasso regression
- `matplotlib/seaborn` - Visualization
- `scipy.linalg` - Matrix operations

**Recommended Reading:**
- Hastie, Tibshirani, & Wainwright (2015) - "Statistical Learning with Sparsity"
- Tibshirani (1996) - Original Lasso paper
- James et al. (2013) - "An Introduction to Statistical Learning"

---

## Tips for Success

1. **Start simple**: Get OLS working first, then add Lasso
2. **Visualize everything**: Plots reveal patterns in your data
3. **Debug your coordinate descent**: Print intermediate values
4. **Verify standardization**: Always check mean ≈ 0, std ≈ 1
5. **Test on the data you know**: Generate synthetic data where you know which features matter
6. **Document assumptions**: Clearly state your choices and reasoning
7. **Compare with sklearn**: After implementing, verify against `sklearn.linear_model.Lasso`

Good luck! This is a comprehensive project that will make you a Lasso expert! 🚀
