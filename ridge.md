# Ridge Regression Problem: Housing Price Prediction

## Problem Statement

You are a data scientist working for a real estate company. You've been given a dataset of housing features and sale prices, but the features are highly correlated (e.g., number of rooms is correlated with square footage, age is correlated with condition). This multicollinearity is causing unstable coefficient estimates in your ordinary least squares (OLS) regression model.

Your task is to implement Ridge regression to create a more stable and reliable predictive model.

---

## Dataset

You have data on 20 houses with the following features:

| House | Square Feet (X₁) | Bedrooms (X₂) | Age (X₃) | Garage Size (X₄) | Price (y) in $1000s |
|-------|------------------|---------------|----------|------------------|---------------------|
| 1     | 1500             | 3             | 10       | 1                | 250                 |
| 2     | 1800             | 3             | 8        | 2                | 290                 |
| 3     | 2400             | 4             | 5        | 2                | 380                 |
| 4     | 2000             | 3             | 12       | 1                | 270                 |
| 5     | 1600             | 2             | 15       | 1                | 220                 |
| 6     | 2200             | 4             | 6        | 2                | 350                 |
| 7     | 1400             | 2             | 20       | 1                | 200                 |
| 8     | 2600             | 4             | 3        | 2                | 420                 |
| 9     | 1900             | 3             | 9        | 2                | 280                 |
| 10    | 2100             | 3             | 7        | 2                | 330                 |
| 11    | 1700             | 3             | 11       | 1                | 240                 |
| 12    | 2300             | 4             | 4        | 2                | 390                 |
| 13    | 1550             | 2             | 18       | 1                | 210                 |
| 14    | 2500             | 4             | 2        | 2                | 410                 |
| 15    | 1850             | 3             | 10       | 1                | 275                 |
| 16    | 2050             | 3             | 8        | 2                | 320                 |
| 17    | 1450             | 2             | 22       | 1                | 190                 |
| 18    | 2150             | 4             | 6        | 2                | 340                 |
| 19    | 1950             | 3             | 9        | 2                | 295                 |
| 20    | 2400             | 4             | 5        | 2                | 385                 |

---

## Tasks

### Part 1: Data Preparation (15 points)

1. **Calculate the correlation matrix** for the features (X₁, X₂, X₃, X₄)
   - Identify which features are highly correlated (|correlation| > 0.7)

2. **Standardize the features** using z-score normalization:
   - For each feature: X̃ⱼ = (Xⱼ - μⱼ) / σⱼ
   - Calculate mean and standard deviation for each feature
   - Show the standardized matrix

3. **Center the response variable** (y):
   - ȳ = mean(y)
   - y_centered = y - ȳ

### Part 2: OLS Regression (10 points)

4. **Compute the OLS solution** (without regularization):
   - β̂_OLS = (X̃ᵀX̃)⁻¹X̃ᵀy
   - Calculate X̃ᵀX̃
   - Find its determinant (what does this tell you about multicollinearity?)
   - Compute the OLS coefficients

### Part 3: Ridge Regression (40 points)

5. **Implement Ridge regression** for λ = 10:
   - β̂_ridge = (X̃ᵀX̃ + λI)⁻¹X̃ᵀy
   - Show the step-by-step calculation
   - Compare coefficients with OLS

6. **Compute predictions and error metrics**:
   - Calculate predictions: ŷ = X̃β̂_ridge + ȳ
   - Compute Mean Squared Error (MSE)
   - Compute R² score

7. **Try different λ values**: {0.1, 1, 10, 100, 1000}
   - Calculate Ridge coefficients for each λ
   - Create a table showing how coefficients change
   - Plot coefficient paths (coefficients vs. log(λ))

### Part 4: Cross-Validation (20 points)

8. **Perform 5-fold cross-validation** to select optimal λ:
   - Test λ values: {0.01, 0.1, 1, 5, 10, 50, 100}
   - For each fold:
     - Split data into training (16 houses) and validation (4 houses)
     - Fit Ridge model on training set
     - Compute validation MSE
   - Calculate average CV MSE for each λ
   - Select λ with minimum CV error

9. **Refit model** with optimal λ on full dataset

### Part 5: Analysis and Interpretation (15 points)

10. **Answer the following questions**:
    - How did Ridge regression address the multicollinearity problem?
    - Which features have the largest coefficients in the Ridge model?
    - How does increasing λ affect the coefficients?
    - What happens to the training MSE as λ increases? Why?
    - Would Lasso be better for this problem? Explain.

11. **Make predictions** for new houses:
    - House A: 2000 sq ft, 3 bedrooms, 7 years old, 2-car garage
    - House B: 1600 sq ft, 2 bedrooms, 15 years old, 1-car garage

---

## Bonus Challenges (Extra 10 points)

12. **Compare Ridge with Elastic Net**: 
    - Implement Elastic Net with α = 0.5 (equal mix of L1 and L2)
    - Compare results

13. **Analytical solution verification**:
    - Verify your Ridge solution by computing the gradient
    - Show that ∇L(β) = 0 at your solution

14. **Implement gradient descent** for Ridge:
    - Compare convergence speed with closed-form solution
    - Plot loss vs. iterations

---

- Compare regularization effects across different λ values

Good luck!
