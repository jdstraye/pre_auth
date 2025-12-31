import json
import pandas as pd
from scripts.analyze_cv_fold_info import summarize_cv_fold_info


def test_summarize_cv_fold_info_basic():
    # Construct a fake search_results dataframe with cv_fold_info
    folds = [
        {'fold': 0, 'smote_enabled': True, 'min_class_count': 10, 'class_counts': [10, 5], 'selected_features': ['a', 'b']},
        {'fold': 1, 'smote_enabled': False, 'min_class_count': 1, 'class_counts': [1, 14], 'selected_features': ['b']},
        {'fold': 2, 'smote_enabled': True, 'min_class_count': 8, 'class_counts': [8, 7], 'selected_features': ['a']},
    ]
    row = {
        'timestamp': 'ts1',
        'model': 'TestModel',
        'mean_f1': 0.75,
        'cv_fold_info': json.dumps(folds)
    }
    df = pd.DataFrame([row])
    summaries = summarize_cv_fold_info(df, top_n=1)
    assert len(summaries) == 1
    s = summaries[0]
    assert s['total_folds'] == 3
    assert s['smote_enabled_count'] == 2
    assert s['folds_min_class_le1'] == 1
    # top selected features should show 'a' and 'b'
    top_feats = dict(s['top_selected_features'])
    assert top_feats.get('a') == 2
    assert top_feats.get('b') == 2
