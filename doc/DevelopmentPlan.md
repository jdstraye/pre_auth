# Predicter Development Plan
# FOCUS - Single Model vs. Ensemble Comparison

## Step 1: Data Preparation
Action: Ensure test_df and train_df are created from a single, initial train_test_split of the full dataset. This step is already captured in 
- train_latest.csv -> /home/jdstraye/proj/shifi/pre_auth.git/data/splits/train_20250821-1520.csv
- test_latest.csv -> /home/jdstraye/proj/shifi/pre_auth.git/data/splits/test_20250821-1520.csv

Code: src/allocate.py


## Step 2: Model Evaluations
Action: Use K-fold Cross-Validation in conjunction with RandomizedSearchCV on train_df to find the optimal hyperparameters for multiple classifiers in both phase 1 and phase 2. 

Code: src/eval_algos.py

Why: This process systematically searches for the best combination of parameters while using K-fold to provide a stable performance estimate, mitigating the risk of overfitting to a specific validation split.

Deliverable: A ranking of the best classifiers to use.

### Classifiers and Hyperparameters evaluated
#### Classifiers
    1. 'RandomForestClassifier': RandomForestClassifier(random_state=42)
    2. 'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    3. 'LGBMClassifier': LGBMClassifier(random_state=42)
    4. 'CatBoostClassifier': CatBoostClassifier(random_state=42, verbose=0)
    5. 'LogisticRegression': LogisticRegression(random_state=42)
    6. 'SVC': SVC(random_state=42)
    7. 'KNeighborsClassifier': KNeighborsClassifier()
#### Hyperparameters
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'XGBClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5],
        'gamma': [0, 0.1, 0.5]
    },
    'LGBMClassifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'num_leaves': [31, 62, 127],
        'learning_rate': [0.01, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    },
    'CatBoostClassifier': {
        'iterations': [100, 200, 300, 400, 500],
        'depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.5],
        'l2_leaf_reg': [1, 3, 5]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    }

### K-Fold Results, Top 100, Round 1, unoptimized labels
1: LGBMClassifier, Params: {'reg_lambda': 0.5, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.7387
2: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 500, 'depth': 3}, Score: 0.7387
3: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.7387
4: LGBMClassifier, Params: {'reg_lambda': 0.5, 'reg_alpha': 0.1, 'num_leaves': 127, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.7355
5: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 400, 'learning_rate': 0.01}, Score: 0.7355
6: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 100, 'depth': 3}, Score: 0.7355
7: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 500, 'depth': 10}, Score: 0.7355
8: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.5}, Score: 0.7355
9: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 500, 'depth': 3}, Score: 0.7355
10: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 400, 'depth': 3}, Score: 0.7355
11: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.7355
12: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 62, 'n_estimators': 400, 'learning_rate': 0.01}, Score: 0.7323
13: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.7323
14: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 500, 'depth': 5}, Score: 0.7323
15: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 500, 'depth': 3}, Score: 0.7323
16: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.7323
17: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.7323
18: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.7323
19: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 300, 'depth': 5}, Score: 0.7323
20: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.7290
21: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.5, 'gamma': 0.5}, Score: 0.7290
22: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.5, 'num_leaves': 127, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.7290
23: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 200, 'depth': 3}, Score: 0.7290
24: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 400, 'depth': 3}, Score: 0.7290
25: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.7290
26: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.7290
27: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.7290
28: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.7290
29: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.7290
30: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': None}, Score: 0.7290
31: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 5, 'iterations': 200, 'depth': 3}, Score: 0.7290
32: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 10}, Score: 0.7290
33: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.5, 'num_leaves': 31, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.7290
34: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.5, 'num_leaves': 62, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.7290
35: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 300, 'depth': 3}, Score: 0.7290
36: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.7258
37: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.7258
38: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}, Score: 0.7258
39: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7258
40: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.7258
41: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.7258
42: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7258
43: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7258
44: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1}, Score: 0.7258
45: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 100, 'depth': 5}, Score: 0.7258
46: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 500, 'depth': 5}, Score: 0.7258
47: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 3}, Score: 0.7258
48: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 400, 'learning_rate': 0.01}, Score: 0.7226
49: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.7226
50: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 400, 'depth': 3}, Score: 0.7226
51: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 200, 'depth': 3}, Score: 0.7226
52: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 200, 'depth': 3}, Score: 0.7226
53: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 100, 'depth': 10}, Score: 0.7226
54: CatBoostClassifier, Params: {'learning_rate': 0.5, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 5}, Score: 0.7226
55: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.7226
56: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7226
57: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.7226
58: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.7226
59: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.7226
60: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.7226
61: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.7226
62: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.7226
63: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.7226
64: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.7226
65: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.7226
66: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.7226
67: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.7226
68: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: 0.7226
69: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 5}, Score: 0.7226
70: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 5}, Score: 0.7226
71: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7226
72: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7226
73: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.7226
74: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.7226
75: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.7226
76: LGBMClassifier, Params: {'reg_lambda': 0.5, 'reg_alpha': 0, 'num_leaves': 127, 'n_estimators': 400, 'learning_rate': 0.01}, Score: 0.7226
77: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7194
78: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.7194
79: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.7194
80: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7194
81: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.7194
82: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.7194
83: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7194
84: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7194
85: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.7194
86: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.5}, Score: 0.7194
87: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 500, 'depth': 3}, Score: 0.7194
88: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 400, 'depth': 10}, Score: 0.7194
89: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 3}, Score: 0.7194
90: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 400, 'depth': 10}, Score: 0.7194
91: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 3}, Score: 0.7194
92: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.7194
93: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.7194
94: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.7194
95: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 5, 'iterations': 200, 'depth': 10}, Score: 0.7194
96: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0.5}, Score: 0.7161
97: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0.5}, Score: 0.7161
98: CatBoostClassifier, Params: {'learning_rate': 0.5, 'l2_leaf_reg': 5, 'iterations': 400, 'depth': 5}, Score: 0.7161
99: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.7161
100: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.5, 'num_leaves': 62, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.7161
1: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4732
2: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4683
3: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4634
4: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4634
5: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4585
6: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4585
7: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': None}, Score: 0.4585
8: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4585
9: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4537
10: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4537
11: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4537
12: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}, Score: 0.4537
13: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4537
14: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4537
15: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4537
16: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4488
17: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4488
18: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4488
19: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4488
20: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4488
21: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4488
22: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4488
23: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4488
24: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4439
25: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.4390
26: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4390
27: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4390
28: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4341
29: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.4341
30: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.4341
31: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.4341
32: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 3}, Score: 0.4341
33: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.4341
34: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.4341
35: RandomForestClassifier, Params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.4341
36: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.4293
37: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 3}, Score: 0.4293
38: RandomForestClassifier, Params: {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4293
39: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.4293
40: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.4293
41: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.4293
42: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 10}, Score: 0.4293
43: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4293
44: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4293
45: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4293
46: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 3}, Score: 0.4293
47: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.4293
48: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': None}, Score: 0.4293
49: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.4244
50: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 5}, Score: 0.4244
51: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.5}, Score: nan
52: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: nan
53: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
54: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
55: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
56: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.1}, Score: nan
57: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0}, Score: nan
58: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
59: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.5}, Score: nan
60: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.5}, Score: nan
61: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
62: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
63: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
64: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0.5}, Score: nan
65: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
66: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
67: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.1}, Score: nan
68: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: nan
69: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan
70: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan
71: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan
72: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
73: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
74: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
75: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0}, Score: nan
76: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
77: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1}, Score: nan
78: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0}, Score: nan
79: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.5, 'gamma': 0.5}, Score: nan
80: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan
81: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0.5}, Score: nan
82: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan
83: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
84: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: nan
85: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: nan
86: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0.1}, Score: nan
87: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
88: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
89: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0.5}, Score: nan
90: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0.1}, Score: nan
91: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
92: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0.5}, Score: nan
93: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
94: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: nan
95: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: nan
96: XGBClassifier, Params: {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: nan
97: XGBClassifier, Params: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.5}, Score: nan
98: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1}, Score: nan
99: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.5, 'gamma': 0}, Score: nan
100: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.5, 'gamma': 0.1}, Score: nan

#### Analysis, Round 1
The accuracy scores of 74% for status and 47% for tier aren't great, but they are 
statistically significant. The status score compared to always guessing 'approved' 
has a p-value of 0.0056. Statistically significant is generally considered 0.05, so 
it's a very significant improvement, though far from infallible.

### K-Fold Results, Top 39, Round 2, improved labels
2025-08-29 13:14:24,567 - __main__ - INFO - Status Model 1: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 100, 'learning_rate': 0.01}, Score: 0.5281
2025-08-29 13:14:24,681 - __main__ - INFO - Status Model 2: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 200, 'depth': 5}, Score: 0.5228
2025-08-29 13:14:25,247 - __main__ - INFO - Status Model 3: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 200, 'depth': 5}, Score: 0.5169
2025-08-29 13:14:25,811 - __main__ - INFO - Status Model 4: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 3}, Score: 0.5168
2025-08-29 13:14:26,098 - __main__ - INFO - Status Model 5: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 3}, Score: 0.5152
2025-08-29 13:14:26,220 - __main__ - INFO - Status Model 6: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: 0.5101
2025-08-29 13:14:26,372 - __main__ - INFO - Status Model 7: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: 0.5083
2025-08-29 13:14:26,616 - __main__ - INFO - Status Model 8: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1}, Score: 0.5070
2025-08-29 13:14:26,862 - __main__ - INFO - Status Model 9: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.5069
2025-08-29 13:14:27,050 - __main__ - INFO - Status Model 10: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.5069
2025-08-29 13:14:27,229 - __main__ - INFO - Status Model 11: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 5}, Score: 0.5059
2025-08-29 13:14:28,017 - __main__ - INFO - Status Model 12: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.1}, Score: 0.5052
2025-08-29 13:14:28,369 - __main__ - INFO - Status Model 13: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.5045
2025-08-29 13:14:28,652 - __main__ - INFO - Status Model 14: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.5039
2025-08-29 13:14:29,016 - __main__ - INFO - Status Model 15: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 5}, Score: 0.5027
2025-08-29 13:14:29,302 - __main__ - INFO - Status Model 16: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.1}, Score: 0.5019
2025-08-29 13:14:29,703 - __main__ - INFO - Status Model 17: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.4993
2025-08-29 13:14:29,828 - __main__ - INFO - Status Model 18: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.4961
2025-08-29 13:14:30,021 - __main__ - INFO - Status Model 19: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: 0.4958
2025-08-29 13:14:30,174 - __main__ - INFO - Status Model 20: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.1}, Score: 0.4950
2025-08-29 13:14:30,474 - __main__ - INFO - Status Model 21: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.4920
2025-08-29 13:14:30,581 - __main__ - INFO - Status Model 22: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.4898
2025-08-29 13:14:30,901 - __main__ - INFO - Status Model 23: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.4868
2025-08-29 13:14:31,345 - __main__ - INFO - Status Model 24: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.4865
2025-08-29 13:14:31,481 - __main__ - INFO - Status Model 25: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4851
2025-08-29 13:14:32,077 - __main__ - INFO - Status Model 26: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.4824
2025-08-29 13:14:32,376 - __main__ - INFO - Status Model 27: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 200, 'learning_rate': 0.1}, Score: 0.4808
2025-08-29 13:14:32,566 - __main__ - INFO - Status Model 28: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.4749
2025-08-29 13:14:33,143 - __main__ - INFO - Status Model 29: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.4731
2025-08-29 13:14:33,245 - __main__ - INFO - Status Model 30: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.4727
2025-08-29 13:14:33,859 - __main__ - INFO - Status Model 31: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4720
2025-08-29 13:14:34,230 - __main__ - INFO - Status Model 32: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 3}, Score: 0.4701
2025-08-29 13:14:34,402 - __main__ - INFO - Status Model 33: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.4695
2025-08-29 13:14:34,604 - __main__ - INFO - Status Model 34: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4589
2025-08-29 13:14:34,985 - __main__ - INFO - Status Model 35: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4566
2025-08-29 13:14:35,557 - __main__ - INFO - Status Model 36: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.1}, Score: 0.4554
2025-08-29 13:14:35,860 - __main__ - INFO - Status Model 37: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.4509
2025-08-29 13:14:36,062 - __main__ - INFO - Status Model 38: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4478
2025-08-29 13:14:36,448 - __main__ - INFO - Status Model 39: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.4453
2025-08-29 13:14:53,716 - __main__ - INFO - Tier Model 1: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 3}, Score: 0.6116
2025-08-29 13:14:53,901 - __main__ - INFO - Tier Model 2: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.5895
2025-08-29 13:14:54,339 - __main__ - INFO - Tier Model 3: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 3}, Score: 0.5862
2025-08-29 13:14:54,928 - __main__ - INFO - Tier Model 4: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 3, 'iterations': 200, 'depth': 5}, Score: 0.5707
2025-08-29 13:14:55,474 - __main__ - INFO - Tier Model 5: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 5}, Score: 0.5698
2025-08-29 13:14:55,817 - __main__ - INFO - Tier Model 6: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': None}, Score: 0.5697
2025-08-29 13:14:56,393 - __main__ - INFO - Tier Model 7: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.5661
2025-08-29 13:14:56,938 - __main__ - INFO - Tier Model 8: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 200, 'depth': 5}, Score: 0.5646
2025-08-29 13:14:57,596 - __main__ - INFO - Tier Model 9: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.5644
2025-08-29 13:14:57,801 - __main__ - INFO - Tier Model 10: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10}, Score: 0.5634
2025-08-29 13:14:58,453 - __main__ - INFO - Tier Model 11: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 3}, Score: 0.5590
2025-08-29 13:14:58,722 - __main__ - INFO - Tier Model 12: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10}, Score: 0.5569
2025-08-29 13:14:59,108 - __main__ - INFO - Tier Model 13: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.5543
2025-08-29 13:14:59,468 - __main__ - INFO - Tier Model 14: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 5}, Score: 0.5526
2025-08-29 13:14:59,658 - __main__ - INFO - Tier Model 15: XGBClassifier, Params: {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.5475
2025-08-29 13:15:00,141 - __main__ - INFO - Tier Model 16: CatBoostClassifier, Params: {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 5}, Score: 0.5459
2025-08-29 13:15:00,948 - __main__ - INFO - Tier Model 17: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1}, Score: 0.5450
2025-08-29 13:15:01,266 - __main__ - INFO - Tier Model 18: RandomForestClassifier, Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.5436
2025-08-29 13:15:01,468 - __main__ - INFO - Tier Model 19: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: 0.5399
2025-08-29 13:15:01,715 - __main__ - INFO - Tier Model 20: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.5392
2025-08-29 13:15:02,094 - __main__ - INFO - Tier Model 21: RandomForestClassifier, Params: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.5389
2025-08-29 13:15:02,641 - __main__ - INFO - Tier Model 22: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.5360
2025-08-29 13:15:02,751 - __main__ - INFO - Tier Model 23: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0}, Score: 0.5314
2025-08-29 13:15:02,904 - __main__ - INFO - Tier Model 24: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.1}, Score: 0.5281
2025-08-29 13:15:03,051 - __main__ - INFO - Tier Model 25: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.5280
2025-08-29 13:15:03,157 - __main__ - INFO - Tier Model 26: RandomForestClassifier, Params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5}, Score: 0.5275
2025-08-29 13:15:03,596 - __main__ - INFO - Tier Model 27: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1}, Score: 0.5271
2025-08-29 13:15:03,797 - __main__ - INFO - Tier Model 28: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.1}, Score: 0.5250
2025-08-29 13:15:04,097 - __main__ - INFO - Tier Model 29: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.5248
2025-08-29 13:15:04,204 - __main__ - INFO - Tier Model 30: XGBClassifier, Params: {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}, Score: 0.5223
2025-08-29 13:15:04,407 - __main__ - INFO - Tier Model 31: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.01}, Score: 0.5203
2025-08-29 13:15:04,711 - __main__ - INFO - Tier Model 32: LGBMClassifier, Params: {'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 200, 'learning_rate': 0.1}, Score: 0.5170
2025-08-29 13:15:04,902 - __main__ - INFO - Tier Model 33: CatBoostClassifier, Params: {'learning_rate': 0.01, 'l2_leaf_reg': 1, 'iterations': 100, 'depth': 3}, Score: 0.5142
2025-08-29 13:15:05,149 - __main__ - INFO - Tier Model 34: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'learning_rate': 0.1}, Score: 0.5120
2025-08-29 13:15:05,458 - __main__ - INFO - Tier Model 35: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 100, 'learning_rate': 0.01}, Score: 0.5119
2025-08-29 13:15:05,568 - __main__ - INFO - Tier Model 36: XGBClassifier, Params: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.1}, Score: 0.5083
2025-08-29 13:15:06,034 - __main__ - INFO - Tier Model 37: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.5020
2025-08-29 13:15:06,209 - __main__ - INFO - Tier Model 38: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 200, 'learning_rate': 0.01}, Score: 0.5020
2025-08-29 13:15:06,390 - __main__ - INFO - Tier Model 39: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.1}, Score: 0.5018

#### Analysis, Round 2
These scores are for F1 instead of simple accuracy. The results are quite encouraging for the #1 models - 
2025-08-29 13:14:24,567 - __main__ - INFO - Status Model 1: LGBMClassifier, Params: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 62, 'n_estimators': 100, 'learning_rate': 0.01}, Score: 0.5281
2025-08-29 13:14:24,651 - __main__ - INFO - Status Classification Report for LGBMClassifier:
2025-08-29 13:14:24,659 - __main__ - INFO - Class 0:
2025-08-29 13:14:24,660 - __main__ - INFO -   Precision: 0.9573
2025-08-29 13:14:24,660 - __main__ - INFO -   Recall: 0.7734
2025-08-29 13:14:24,660 - __main__ - INFO -   F1-score: 0.8556
2025-08-29 13:14:24,660 - __main__ - INFO -   Support: 203.0
2025-08-29 13:14:24,660 - __main__ - INFO - Class 1:
2025-08-29 13:14:24,660 - __main__ - INFO -   Precision: 0.7000
2025-08-29 13:14:24,660 - __main__ - INFO -   Recall: 0.8116
2025-08-29 13:14:24,661 - __main__ - INFO -   F1-score: 0.7517
2025-08-29 13:14:24,661 - __main__ - INFO -   Support: 69.0
2025-08-29 13:14:24,661 - __main__ - INFO - Class 2:
2025-08-29 13:14:24,661 - __main__ - INFO -   Precision: 0.4918
2025-08-29 13:14:24,662 - __main__ - INFO -   Recall: 0.9091
2025-08-29 13:14:24,662 - __main__ - INFO -   F1-score: 0.6383
2025-08-29 13:14:24,662 - __main__ - INFO -   Support: 33.0
2025-08-29 13:14:24,662 - __main__ - INFO - Class macro avg:
2025-08-29 13:14:24,663 - __main__ - INFO -   Precision: 0.7164
2025-08-29 13:14:24,663 - __main__ - INFO -   Recall: 0.8314
2025-08-29 13:14:24,663 - __main__ - INFO -   F1-score: 0.7485
2025-08-29 13:14:24,663 - __main__ - INFO -   Support: 305.0
2025-08-29 13:14:24,663 - __main__ - INFO - Class weighted avg:
2025-08-29 13:14:24,663 - __main__ - INFO -   Precision: 0.8487
2025-08-29 13:14:24,663 - __main__ - INFO -   Recall: 0.7967
2025-08-29 13:14:24,663 - __main__ - INFO -   F1-score: 0.8086
2025-08-29 13:14:24,663 - __main__ - INFO -   Support: 305.0
2025-08-29 13:14:24,663 - __main__ - INFO - Macro Average Metrics:
2025-08-29 13:14:24,664 - __main__ - INFO -   Precision: 0.7164
2025-08-29 13:14:24,664 - __main__ - INFO -   Recall: 0.8314
2025-08-29 13:14:24,664 - __main__ - INFO -   F1-score: 0.7485
2025-08-29 13:14:24,664 - __main__ - INFO - Weighted Average Metrics:
2025-08-29 13:14:24,664 - __main__ - INFO -   Precision: 0.8487
2025-08-29 13:14:24,664 - __main__ - INFO -   Recall: 0.7967
2025-08-29 13:14:24,664 - __main__ - INFO -   F1-score: 0.8086
2025-08-29 13:14:24,666 - __main__ - INFO - Saved Status classification report to logs/status_classification_report.csv
2025-08-29 13:14:24,667 - __main__ - INFO - Feature importances:
2025-08-29 13:14:24,667 - __main__ - INFO - final_contract_amount: 656.0000
2025-08-29 13:14:24,667 - __main__ - INFO - DebtToIncome: 485.0000
2025-08-29 13:14:24,667 - __main__ - INFO - AutomaticFinancing_Amount: 376.0000
2025-08-29 13:14:24,667 - __main__ - INFO - DebtResolution_Amount: 368.0000
2025-08-29 13:14:24,667 - __main__ - INFO - AutomaticFinancing_Score: 350.0000
2025-08-29 13:14:24,667 - __main__ - INFO - DebtResolution_Score: 317.0000
2025-08-29 13:14:24,667 - __main__ - INFO - 0UnsecuredFunding_Amount: 240.0000
2025-08-29 13:14:24,667 - __main__ - INFO - DebtResolution_DebtToIncome: 146.0000
2025-08-29 13:14:24,667 - __main__ - INFO - 0UnsecuredFunding_status_Declined: 89.0000
2025-08-29 13:14:24,667 - __main__ - INFO - 0UnsecuredFunding_Score: 80.0000
2025-08-29 13:14:24,667 - __main__ - INFO - 0UnsecuredFunding_DebtToIncome: 74.0000
2025-08-29 13:14:24,668 - __main__ - INFO - DebtResolution_below_600_: 60.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_Status_Approved: 31.0000
2025-08-29 13:14:24,668 - __main__ - INFO - 0UnsecuredFunding_PayD: 18.0000
2025-08-29 13:14:24,668 - __main__ - INFO - 0UnsecuredFunding_status_If_Fixed: 12.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_Status_Declined: 6.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_below_600_: 2.0000
2025-08-29 13:14:24,668 - __main__ - INFO - DebtResolution_missing_: 2.0000
2025-08-29 13:14:24,668 - __main__ - INFO - user_initials_label: 0.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_missing_: 0.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_Status_NA_: 0.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_Details_in_the_wallet_: 0.0000
2025-08-29 13:14:24,668 - __main__ - INFO - AutomaticFinancing_Details_just_available_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - AutomaticFinancing_Details_NA_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - AutomaticFinancing_DebtToIncome: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_missing_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_below_600_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_status_As_Is: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_status_NA_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_Details_To_book_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_Details_just_CL_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_Details_NA_: 0.0000
2025-08-29 13:14:24,669 - __main__ - INFO - 0UnsecuredFunding_Collections: 0.0000
2025-08-29 13:14:24,670 - __main__ - INFO - DebtResolution_score_missing_: 0.0000

So the status accuracy improved from 74% to 85%. More importantly, the **precision** of approved is really good - 95%. This means that when classified as approved, the true status is 'approved' 95% of the time. Likewise, the **recall** is 77%. This means that of all the approved applicants, we identify 77% of them.  (how many classified as X are really X)

#### Next steps after Round 2
'Rejected' actually means that the applicant was approved but later decided to self-fund. So, 'rejected' is actually the same class as approved.


## Step 3: Ensemble Creation
Action: Use the above results to create 2 ensembles - a simple voter ensemble 
of the top 3 and a cascaded feedback where Phase 2 probabilities are used as an
input to Phase 1.
The results should be pretty (ReactUI/html) and have associated probabilities.

Code: src/finalize.py

Why: This approach creates an ensemble of diverse models, each trained on a slightly different subset of the data. Their combined predictions will be more robust than any single model's.

## Step 4: Final Evaluation and Comparison
Action: Evaluate both the Best model from evaluations and the ensemble on the unseen test_df.

Code: src/final_eval.py

Why: This is the ultimate, unbiased comparison. It tells whether the extra complexity of the ensemble approach provides a significant performance gain over a well-optimized single model.

Deliverable: A detailed report of the performance metrics for both models on test_df.

## Step 5: Analyze the best model
Action: Do a thorough analysis of the best/final model that will be delivered. This might be a K-fold evaluations on the entire data set but using just this final model instead of creating a new one for every fold. Confusion matrix should be included. 
Significant documnentation and graphs are expected.

Code: tests/characterize.py

Why: This is a full understanding of what is being delivered and what its warts are.

Deliverable: A detailed report of the chosen classifier.

## Step 6: Package for delivery
Action: Use pyparser to create a final executable as the ultimate deliverable. There should be good user interface with json file input.

Code: 

Why: For customer ease-of-use.

Deliverable: PreAuth.exe