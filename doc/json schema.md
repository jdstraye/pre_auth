The schema for the column_header.json is complicated:

a. LABEL - Auto-generated
"labels":"auto-generated", "generation_methods":"[<methods to generate the labels>]". The labels should be auto-generated. "generation_methods" types -
- "simple" - use LabelEncoder for a direct 1:1 mapping. This is fine for targets (Y) but should be avoided for features (X).
- "frequency" - use the number of times the element appears or +/- as few as possible to maintain unique mapping.
- "grouped" - Start with a 2% limit. Any element with fewer occurrences than 2% of the total unique number of elements should be grouped together into an "other" group. The methods will be applied in the order they are named.
Ex: There are over 150 unique initials. They don't have to be chosen with care nor one-hot encoded because they are not expected to be critical and would cause a dimensionality explosion. Since we don't want to imply ordinal priority, we don't want a simple mapping with LabelEncoder, though, either. 
{
    "name": "user_initials",
    "categorical": "True",
    "X": "False",
    "Y": "False",
    "labels":"auto-generated",
    "generation_methods": ["grouped", "frequency"]
  },
  Therefore, we want to auto-generate labels that will first put all user_initials that appear less than 3 times (2% of 150 = 3) into an "other" category. Then, we generate the rest of the labels based on frequency of occurrence.
  {
    "name": "user_initials_label",
    "categorical": "True",
    "X": "True",
    "Y": "False",
    "labels":"auto-generated"
  },

b. LABEL Source
"labels_from":"<another column that is the source where the labels are defined>" Generally, this is in the column/feature named with an "_label" appendage. This column/feature will contain the labels whereas the original column retains the initial values.
Ex:
  {
    "name": "user_initials_label",
    "categorical": "True",
    "X": "True",
    "Y": "False",
    "labels_from": "user_initials"
  },
This puts the auto-generated labels from 'user_initials' into the column/feature named 'user_initials_label.'

c. LABEL - Custom
"labels": {"<original_value>":"<label>"} For hand-coding the labels, when the raw data is <original_value>, use the label, <label> instead.
Ex: 
  {
    "name": "final_contract_status",
    "categorical": "True",
    "X": "False",
    "Y": "False",
    "labels":{
      "approved": "0",
      "declined": "1",
      "rejected": "2"
    }
  },
The input has text strings - "approved", "declined", "rejected" - that need to be labeled as 0,1,2 respectively. This is a target, so LabelEncoder could have been used, but hard-coding keeps the labels consistent from run-to-run.

d. LABEL Description
"label_descriptions":["<description for the first label>","<description for the 2nd label>", et. al.] At present, this is for documentation purposes for complex label schemes.
Ex:
  {
    "name": "final_contract_tier",
    "categorical": "True",
    "X": "False",
    "Y": "False",
    "labels":{  
      "0.095": "0",
      "0.165": "0",
      "0.045": "0",
      "0.24": "1",
      "0.31": "1",
      "0.34": "2",
      "0.41": "2",
      "0.44": "3",
      "0.51": "3",
      "0.12": "0",
      "NA": "-1"
    },
    "label_descriptions":["0-Tier1","1-Tier2","2-Tier3","3-Tier4","-1:not approved"]
  }
e. MAP
"mapped_from":"another column/feature that is the source where the mapping is defined>" "mapping":{"<raw value>>":"<desired value to be presented in this column>". Similar to labels, the "labels" or "mapping" key will be in the source feature/column while the derived column/feature will have the "labels_from" or "mapped_from" key.
Example with both "mapping" and "labels" that will show up in 2 separate columns/features:
  {
    "name": "final_contract_tier",
    "categorical": "True",
    "X": "False",
    "Y": "False",
    "labels":{  
      "0.095": "0",
      "0.165": "0",
      "0.045": "0",
      "0.24": "1",
      "0.31": "1",
      "0.34": "2",
      "0.41": "2",
      "0.44": "3",
      "0.51": "3",
      "0.12": "0",
      "NA": "-1"
    },
    "label_descriptions":["0-Tier1","1-Tier2","2-Tier3","3-Tier4"],
    "mapping":{
      "0.095": "A - Tier 1",
      "0.165": "A - Tier 1 Low APR",
      "0.045": "A - Tier 1 VIP",
      "0.24": "A - Tier 2",
      "0.31": "A - Tier 2 Low APR",
      "0.34": "A - Tier 3",
      "0.41": "A - Tier 3 Low APR",
      "0.44": "A - Tier 4",
      "0.51": "A - Tier 4 Low APR",
      "0.12": "B - Tier 1 Prime",
      "NA": "error/TBD/not approved"
    }
  },
  {
    "name": "final_contract_tier_name",
    "categorical": "True",
    "X": "False",
    "Y": "False",
    "mapped_from": "final_contract_tier"
  },
  {
    "name": "final_contract_tier_label",
    "categorical": "False",
    "X": "False",
    "Y": "True",
    "labeled_from": "final_contract_tier"
  },
final_contract_tier is the original feature/column from the source data, and it is used to generate 2 new features/columns - final_contract_tier_label and final_contract_tier_name. All the labels, label documentation, and mapping are defined in the original raw data column/feature so that these derivative columns/features are simply instantiated with a reference to how they should be derived.
f. One-hot Encoding (OHE). OHE is the most important method for describing important features (X).
- "ohe":{"<raw value from source>": "<binary value column>"}
Ex: an input takes one of 3 values - "As Is", "If Fixed", and "Declined". OHE means that there will be 3 columns - "As Is", "If Fixed", and "Declined." Where an entry is "As Is", the column named "As Is" should have a '1' while the other columns will have '0'.
  {
    "name": "0_Unsecured_Funding_Status",
    "categorical": "True",
    "X": "True",
    "Y": "False",
    "ohe": {
      "As Is": "0_Unsecured_Funding_status_As_Is",
      "If Fixed": "0_Unsecured_Funding_status_If_Fixed",
      "Declined": "0_Unsecured_Funding_status_Declined"
    }
  }
- "ohe_from":"<the column where ohe is defined>"
- "ohe_key":"<the key in the ohe set that corresponds to this column>"
Ex:
  {"name": "0_Unsecured_Funding_status_As_Is",
    "X":"True",
    "Y":"False",
    "ohe_from":"0_Unsecured_Funding_Status"
    "ohe_key":"As is"
  },
  {"name": "0_Unsecured_Funding_status_If_Fixed",
    "X":"True",
    "Y":"False",
    "ohe_from":"0_Unsecured_Funding_Status"
    "ohe_key":"If Fixed"
  },
  {"name": "0_Unsecured_Funding_status_Declined",
    "X":"True",
    "Y":"False",
    "ohe_from":"0_Unsecured_Funding_Status"
    "ohe_key":"Declined"
  },
The 4 entries above fully describes a One-hot encoding for a column with 3 options.
