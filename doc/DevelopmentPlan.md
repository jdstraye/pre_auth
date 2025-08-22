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

### Results, Top 100

## Step 3: Ensemble Creation
Action: Use the above results to create a simple voter ensemble of the top 3. The results should be pretty (ReactUI/html) and have associated probabilities.

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