# 5243_proj1

## Overview

This project examines how occupation-level working conditions are associated with individual sleep duration. We combine employee survey data with U.S. Bureau of Labor Statistics (BLS) Occupational Requirements Survey (ORS) data to construct structured stress-related indicators.

## Data Sources

### 1. Employee Dataset (Kaggle)

The employee survey dataset provides:
- Demographics (age, gender, education)
- Job characteristics (occupation, job level, employment type)
- Self-reported measures (workload, stress, work–life balance)
- Outcome variable: `SleepHours`

### 2. BLS Occupational Requirements Survey (ORS)

The ORS provides occupation-level percentage measures related to:
- Work schedule variability
- Ability to pause work
- Work pace (fast / slow / varying)

## Feature Engineering

Because ORS variables are reported as **percentages** and are often **bounded** (0–100) or **compositional** (pace shares sum to 100), we apply the following transformations to obtain usable continuous predictors.

### 1) Percentage → Proportion

All ORS percentage variables are converted to proportions:
- `p = Percent / 100`

This rescales values into \([0,1]\).

### 2) Logit Transformation (Bounded Shares)

For bounded variables such as schedule variability and pause ability, we apply a logit transform:

\[
\log\left(\frac{p}{1-p}\right)
\]

This helps remove boundary constraints and improves suitability for regression modeling.

Constructed variables:
- `work_schedule_variability` (and optionally its logit index)
- `pause_work` (and optionally its logit index)

### 3) Log-Ratio + PCA (Work Pace)

Work pace indicators are compositional:
- Percent fast pace
- Percent slow pace
- Percent varying pace

Processing steps:
1. Convert to proportions.
2. Apply log-ratio transformations to remove the sum-to-one constraint.
3. Use PCA to synthesize correlated pace indicators into a single factor (first principal component).

Constructed variable:
- `work_pace` (or `work_pace_pca`)

## Final Stress-Related Variables

The following engineered occupation-level predictors are used in downstream analysis:

- `work_schedule_variability`  
- `pause_work`  
- `work_pace`  

These represent three key dimensions of occupational stress:
- **Schedule instability**
- **Work autonomy**
- **Work intensity**

## Dataset

- Final merged dataset: **3,025 observations**
- Individual-level survey data merged with occupation-level ORS indicators via occupation matching.

## Goal

Estimate the association between structured occupational working-condition indicators and sleep duration, controlling for demographic and job-related covariates.
