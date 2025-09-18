"""
Integration tests for the complete ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

# Add src to path for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline_coordinator import MLPipelineCoordinator
from components.smote_sampler import MaybeSMOTESampler
from components.feature_selector import FeatureSelector, FeatureSelectingClassifier

class TestPipelineIntegration:
    """Integration tests for the complete ML pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=10,
            n_classes=2,
            weights=[0.8, 0.2],  # Imbalanced
            random_state=42
        )
        
        # Create DataFrame with mixed feature types
        columns = [f'numeric_{i}' for i in range(15)] + [f'category_{i}' for i in range(5)]
        X_df = pd.DataFrame(X, columns=columns)
        
        # Make last 5 columns categorical (binary)
        for col in [f'category_{i}' for i in range(5)]:
            X_df[col] = (X_df[col] > X_df[col].median()).astype(int)
        
        return X_df, y
    
    @pytest.fixture
    def coordinator(self):
        """Create pipeline coordinator."""
        return MLPipelineCoordinator(
            enable_debugging=True,
            export_debug_info=False  # Don't create files during testing
        )
    
    def test_complete_pipeline_creation(self, coordinator, sample_data):
        """Test creating a complete pipeline."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        categorical_features = [f'category_{i}' for i in range(5)]
        
        pipeline = coordinator.create_safe_pipeline(
            base_estimator=base_estimator,
            categorical_features=categorical_features
        )
        
        # Check pipeline structure
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'smote'
        assert pipeline.steps[1][0] == 'feature_selecting_classifier'
        
        # Check component types
        assert isinstance(pipeline.steps[0][1], MaybeSMOTESampler)
        assert isinstance(pipeline.steps[1][1], FeatureSelectingClassifier)
    
    def test_pipeline_fitting(self, coordinator, sample_data):
        """Test fitting the complete pipeline."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        categorical_features = [f'category_{i}' for i in range(5)]
        
        pipeline = coordinator.create_safe_pipeline(
            base_estimator=base_estimator,
            categorical_features=categorical_features
        )
        
        # Fit pipeline
        fitted_pipeline = coordinator.fit_pipeline(pipeline, X_df, y)
        
        # Check that components are fitted
        assert hasattr(fitted_pipeline.steps[0][1], '_validator')  # SMOTE sampler
        assert fitted_pipeline.steps[1][1].selected_features_ is not None  # Feature selector
    
    def test_pipeline_prediction(self, coordinator, sample_data):
        """Test making predictions with fitted pipeline."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        categorical_features = [f'category_{i}' for i in range(5)]
        
        pipeline = coordinator.create_safe_pipeline(
            base_estimator=base_estimator,
            categorical_features=categorical_features
        )
        
        # Fit and predict
        fitted_pipeline = coordinator.fit_pipeline(pipeline, X_df, y)
        predictions = fitted_pipeline.predict(X_df)
        probabilities = fitted_pipeline.predict_proba(X_df)
        
        # Validate predictions
        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_cross_validation(self, coordinator, sample_data):
        """Test cross-validation with the complete pipeline."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        categorical_features = [f'category_{i}' for i in range(5)]
        
        pipeline = coordinator.create_safe_pipeline(
            base_estimator=base_estimator,
            categorical_features=categorical_features
        )
        
        # Cross-validate
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = coordinator.cross_validate_pipeline(
            pipeline, X_df, y, cv=cv, scoring='f1_macro'
        )
        
        # Validate results
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert 0 <= cv_results['mean_score'] <= 1
    
    def test_input_validation_catches_problems(self, coordinator):
        """Test that input validation catches data problems."""
        # Create problematic data
        X_bad = pd.DataFrame({'col1': [1, 2, np.nan, np.inf]})
        y_bad = np.array([0, 1, 0, 1])
        
        # Should catch data problems
        validation_report = coordinator.validate_pipeline_input(X_bad, y_bad)
        
        assert not validation_report["passed"]
        assert len(validation_report["issues"]) > 0
    
    def test_pipeline_with_disabled_smote(self, coordinator, sample_data):
        """Test pipeline with SMOTE disabled."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        
        smote_config = {'enabled': False}
        feature_config = {'max_features': 10}
        
        pipeline = coordinator.create_pipeline(
            base_estimator=base_estimator,
            smote_config=smote_config,
            feature_selection_config=feature_config
        )
        
        # Should work without SMOTE
        fitted_pipeline = coordinator.fit_pipeline(pipeline, X_df, y)
        predictions = fitted_pipeline.predict(X_df)
        
        assert len(predictions) == len(y)
    
    def test_pipeline_parameter_setting(self, coordinator, sample_data):
        """Test setting pipeline parameters."""
        X_df, y = sample_data
        
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        
        pipeline = coordinator.create_safe_pipeline(base_estimator=base_estimator)
        
        # Set parameters
        pipeline.set_params(
            smote__k_neighbors=5,
            feature_selecting_classifier__max_features=8,
            feature_selecting_classifier__estimator__n_estimators=20
        )
        
        # Check parameters were set
        assert pipeline.get_params()['smote__k_neighbors'] == 5
        assert pipeline.get_params()['feature_selecting_classifier__max_features'] == 8
        assert pipeline.get_params()['feature_selecting_classifier__estimator__n_estimators'] == 20
    
    def test_get_pipeline_summary(self, coordinator):
        """Test getting pipeline summary."""
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        
        pipeline = coordinator.create_safe_pipeline(base_estimator=base_estimator)
        summary = coordinator.get_pipeline_summary(pipeline)
        
        assert 'steps' in summary
        assert len(summary['steps']) == 2
        assert summary['total_parameters'] > 0
        assert summary['steps'][0]['name'] == 'smote'
        assert summary['steps'][1]['name'] == 'feature_selecting_classifier'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
