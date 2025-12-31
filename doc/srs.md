# Overview
The goal is to output a classifier with >90% f1 score on the test data, so we need to thoroughly search classifier types and hyperparameters to get there.

# Implementation
The high level execution is run_intensive_search, and should be
ingest -> allocate -> pipeline_coordinator (using src.components.smote_sampler and src.components.feature_selector so that we can refine down to the most important features). The output should include all relevant metrics of a classifier (e.g., accuracy, precision, recall, f1) and feature importance so that it's clear which features are working and which are superfluous for the purpose of furhter refinement.