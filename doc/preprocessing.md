# Introduction
This document contains the pre-processing that must be done before the model can be run on the data.
## Normalization
To normalize the data, convert every number to plain values, eg
- $95,000 -> 95000
- $0 -> 0

## An Explanation of the 'Offers' field
There are 2 or 3 offers for every record. The first offer, Automatic Financing, is generally in good condition, but the others can need quite a bit of work. The 3 are 
- "Automatic Financing"
- "0% Unsecured Funding"
- "Debt Resolution"

## Imputing and Binning/One-hot Encoding
There are 2 primary data types for the fields/columns: numeric and categorical. Numeric would be things like Score and DebtToIncome. Categorical would be most of the others. Categorical and even some numeric benefit from binning and one-hot encoding. For example, Scores has an established demarcation line at 600. Therefore, Scores has its numeric value, but it also warrants a separate column flagging if the Score is below 600, '_below_600?'. Another example comes in the form of Missing Not At Random (MNAR) information. For example, 'Debt Resolution' offers do not include Score if but only if the offer is declined. Therefore, the blank score is imputed a value of -999 and an extra field is added - '_score_missing?'. 'Debt Reolution' offers also blank out the 'DebtToIncome' field of 'Declined' offers, so they are also imputed -999.

## New Columns/Fields
In order for the model to train quickly to the right parameters, some of the data must be imputed and other features must be highlighted. To stress some of the features to the model, the following columns/fields must be added:
- 'DebtResolution_score_missing?'
- '0UnsecuredFunding_Contingencies'
- '0UnsecuredFunding_Collections'
- 'AutomaticFinancing_below_600?'
- '0UnsecuredFunding_below_600?'
- 'DebtResolution_below_600?'
- '0UnsecuredFunding_missing?'

## Deleted Columns/Fields
Other fields in the JSON did not have any distinguishing data and were removed:
- 'DebtResolution_Details'
- 'DebtResolution_Contingencies'
- 'AutomaticFinancing_Contingencies'

## Offer anomalies
### Empty Data for Rejections
Both '0% Unsecured Funding' and 'Debt Resolution' offers frequently leave blanks if the offer_status is not 'Approved' for 'Debt Resolution' or 'As Is' for '0% Unsecured Funding'. Any such blanks in 'status' should be made 'Declined' and any blanks in amount should be set to $0

### Missing Offers
Some records don't have all 3 offers. To deal with this, we need to create new columns and impute the data that should be there. The new columns are '0UnsecuredFunding_missing?' and 'DebtResolution_missing?' Where the respective offer is missing, the value in the column should be '1' otherwise '0'. Any missing offers should be filled in with
Name: Debt Resolution | 0% Unsecured Funding
Score: -1
_below_600?: -1
Status: -1
Amount: -1
Details: '' for 'Debt Resolution', NA for '0% Unsecured Funding'
Contingencies: '' for 'Debt Resolution', -1 for '0% Unsecured Funding'
DebtToIncome: -1

### Contingencies
'0% Unsecured Funding' has special information in its Contingencies field. This information can be split into 2 columns. The special information is of the form 'PayD-$<value1>' or 'PayD-$<value1>,Collections-$<value2>'
Therefore, we create 2 columns - '0UnsecuredFunding_PayD' and '0UnsecuredFunding_Collections'
0UnsecuredFunding_PayD gets value1 and 0UnsecuredFunding_Collections gets value2.

## Numerical Columns with Buckets
### Below 600 scores
For entries that are marked "below 600", we change that to a numeric value of 599 and also add a new field called "_below_600?". All scores below 600 and the ones labeled 'Below 600' get a '1' in this new column. Otherwise, they get a '0'.
Since the scores are associated with the offers, you are adding 3 columns - 'AutomaticFinancing_below_600?', '0UnsecuredFunding_below_600?', and 'DebtResolution_below_600?'

## Missing Not At Random (MNAR)
### Debt Resolution Offers
Debt Resolution hides the score if the offer is Declined but provides it if Approved. For these cases, fill in the score with -999 and create a new column - 'DebtResolution_score_missing?' and populate it with '1' if the score is missing, '0' otherwise.

## Labels
### The incoming tier data is already a decimal value, but we need whole numbers
As a decimal, the model is more likely to ascribe a numeric weight, e.g., 0.90 is twice as good as 0.45. Instead, we will use a lookup table of:
|Original decimal | Tier Name          | New Label|
|-----------------|--------------------|-----
-----|
|0.045            | A - Tier 1 VIP     |      0 |
|0.095            | A - Tier 1         |      1 |
|0.12             | B - Tier 1 Prime   |      2 |
|0.165            | A - Tier 1 Low APR |      3 |
|0.24             | A - Tier 2         |      4 |
|0.31             | A - Tier 2 Low APR |      5 |
|0.34             | A - Tier 3         |      6 |
|0.41             | A - Tier 3 Low APR |      7 |
|0.44             | A - Tier 4         |      8 |
|0.51             | A - Tier 4 Low APR |      9 |
|empty            | null (declined)    |     99 |
### final_contract_status: 4 values
approved, declined, rejected, refunded. It's tempting to merge approved&refunded and declined&rejected and then one-hot encode the remaining 2. However, a machine learning model might be able to find subtle differences in the input data that lead to these outcomes. By keeping all four as separate categories, we allow the model to learn these distinctions. For a target variable with four values, the best approach is Label Encoding:
|Original Description| Label|
|--------------------|------|
| approved           | 0    |
| declined           | 1    |
| rejected           | 2    |
| refunded           | 3    |

## One-hot Encoding
For input features with a low number of distinct categories, one-hot encoding was used to avoid implying a false ordinal relationship. Specifically, it was applied to 
- AutomaticFinancing_Status ->
  - AutomaticFinancing_Status_Approved?
  - AutomaticFinancing_Status_Declined?
- AutomaticFinancing_Details -->
  - AutomaticFinancing_Details_in_the_wallet?
  - AutomaticFinancing_Details_just_available?
- 0UnsecuredFunding_Status -->
  - 0UnsecuredFunding_Status_As_is?
  - 0UnsecuredFunding_Status_Declined?
  - 0UnsecuredFunding_Status_if_Fixed?
  - 0UnsecuredFunding_Status_NA?
- 0UnsecuredFunding_Details -->
  - 0UnsecuredFunding_Details_To_book?
  - 0UnsecuredFunding_Details_just_CL?
  - 0UnsecuredFunding_Details_NA?
- DebtResolution_Status -->
  - DebtResolution_Status_Approved?
  - DebtResolution_Status_Declined?
  - DebtResolution_Status_NA?

## Output
Google Gemini Prompt for implementing import.py -
I need help importing and preprocessing json data for a classifier. The current import code is
<import.py code>
The final df should be written to data/prefi_weaviate_clean-1_import_processed.csv.

What needs done because it has heretofore been done manually:
A) Normalization. To normalize the data, I convert every number to plain values, e.g.
- $95,000 -> 95000
- $0 -> 0
B) 'user_initials' should be labeled
C) final_contract_status should be labeled and become 'final_contract_status_label'.
If there are multiple contracts, use the most recent one based on the 'created_at' date field. 
The current distribution is
|Class | Samples | Proportion |
|------|---------|------------|
| 0    |  205    | ~66.1%     |
| 1    |   70    | ~22.6%     |
| 2    |   34    | ~11.0%     |
| 3    |    1    |  ~0.3%     |
D) All Approved applications are also given a 'tier'. These also need labeled, even 
thought the field appears to be numeric. The final column should be 'final_contract_tier_label'.
Raw distribution:
| Class| Samples  |   Proportion|
|------|----------|-------------|
| 0    |  2       |   ~1.0%     |
| 1    | 68       |    ~33.2%   |
| 2    |  3       |   ~1.5%     |
| 3    |  2       |   ~1.0%     |
| 4    | 64       |    ~31.2%   |
| 5    |  2       |   ~1.0%     |
| 6    | 46       |    ~22.4%   |
| 7    |  1       |   ~0.5%     |
| 8    | 14       |    ~6.8%    |
| 9    |  1       |   ~0.5%     |
| 10   |  2       |   ~1.0%     |

E) Those, 'final_contract_status_label' and 'final_contract_tier_label', are the primary
Y variables. The rest of the fields are X (input) variables. By far, most of those inputs 
are in the 'Offers' cluster. There should be 3 offers:
- 'Automatic Financing'
- '0% Unsecured Funding'
- 'Debt Resolution'
If an Offer is missing, I create a field/column _missing?, i.e.:
- 'AutomaticFinancing_missing?'
- '0UnsecuredFunding_missing?'
- 'Debt Resolution_missing?'
and set that field to 1 with all the others being 0.
If the offer is missing, I also impute the following:
Score: -1
_below_600?: -1
Status: NA
Amount: -1
Details: NA
Contingencies: 'NA' for 'Debt Resolution', -1 in both columns for '0% Unsecured Funding'
DebtToIncome: -1
F) The following fields should be made one-hot encoded:
'AutomaticFinancing_Details' -> 'AutomaticFinancing_Details_in_the_wallet?' and 'AutomaticFinancing_Details_just_available?'
'0UnsecuredFunding_Status' -> '0UnsecuredFunding_Status_As_is?',	'0UnsecuredFunding_Status_Declined?', '0UnsecuredFunding_Status_if_Fixed?', and '0UnsecuredFunding_Status_NA?'
'0UnsecuredFunding_Details' --> '0UnsecuredFunding_Details_To_book?', '0UnsecuredFunding_Details_just_CL?', and '0UnsecuredFunding_Details_NA?'
G) '0UnsecuredFunding_Contingencies' should be split into 2 columns. An example 
interesting value is "PayD-$20,000,Collections-$5,700". Therefore, 
'0UnsecuredFunding_PayD' should default to 0 but be given a value 
extracted from the string in the field - e.g. value1 in PayD-$<value1>, 
if there is one. '0UnsecuredFunding_Collections' should be the Collections 
part of the string, if there is one - e.g., <value2> in 
PayD-$[,0-9]+,Collections-$<value2>.
H) Empty Data for Rejections. Both '0UnsecuredFunding' and 'Debt Resolution' offers
frequently leave blanks if the offer_status is not 'Approved' for 'Debt Resolution'
or 'As Is' for '0% Unsecured Funding'. Any such blanks in 'status' should be made
'Declined' and any blanks in amount should be set to $0.
I) Some offer scores just have a string "below 600". Change that to a numeric value
of 599 and also add a new field called "_below_600?". All scores below 600 and the 
ones labeled 'Below 600' get a '1' in this new column. Otherwise, they get a '0'.
J) Hidden scores. 'Debt Resolution' hides the score if the offer is Declined but 
provides it if Approved. For these cases, fill in the score with -999 and create 
a new column - 'Debt Resolution_score_missing?' and populate it with '1' if the 
score is missing, '0' otherwise.
K) Summary
If the result were implemented in a table, it would have the following columns:
Input/Output/Not Needed - Column Name
I have a new import.py, but the input columns have changed -
All the columns and their label (x or y) - 
NA-record_id
NA-user_initials
X-user_initials_label
X-DebtToIncome
NA-final_contract_status
Y-final_contract_status_label
NA-final_contract_tier
NA-final_contract_tier_name
Y-final_contract_tier_label
X-final_contract_amount
X-AutomaticFinancing_Score
X-AutomaticFinancing_below_600?
NA-AutomaticFinancing_Status
X-AutomaticFinancing_Amount
NA-AutomaticFinancing_Details
X-AutomaticFinancing_Details_in_the_wallet?
X-AutomaticFinancing_Details_just_available?
X-AutomaticFinancing_DebtToIncome
X-0UnsecuredFunding_missing?
X-0UnsecuredFunding_Score
X-0UnsecuredFunding_below_600?
X-0UnsecuredFunding_Status
X-0UnsecuredFunding_Amount
NA-0UnsecuredFunding_Details
X-0UnsecuredFunding_PayD
X-0UnsecuredFunding_Collections
NA-0UnsecuredFunding_Contingencies
X-0UnsecuredFunding_DebtToIncome
X-DebtResolution_missing?
X-DebtResolution_Score
X-DebtResolution_below_600?
NA-DebtResolution_Status
X-DebtResolution_Amount
X-DebtResolution_DebtToIncome
NA-DebtResolution_Details
X-Automatic_Financing_Status_Approved
X-Automatic_Financing_Status_Declined
X-0%_Unsecured_Funding_Status_As Is
X-0%_Unsecured_Funding_Status_Declined
X-0%_Unsecured_Funding_Status_If Fixed
X-0%_Unsecured_Funding_Status_Missing
X-Debt_Resolution_Status_Approved
X-Debt_Resolution_Status_Missing
X-0UnsecuredFunding_Details_NA
X-0UnsecuredFunding_Details_Unsecured Revolving Credit Lines - 0% for 12-21 months
X-0UnsecuredFunding_Details_Unsecured Revolving Credit Lines - 0% for 12-21 months - To book: [URL_REMOVED]

More information on the import.py output:
#### Y (Target Columns):
- final_contract_status_label: Target for Phase 1 (status classification, 4 classes: 0=approved, 1, 2, 3; severe imbalance with class 3 having 1 sample).
- final_contract_tier_label: Target for Phase 2 (tier classification, 11 classes, filtered to approved contracts; many rare classes, e.g., 7, 9 with 1 sample each).


#### X (Feature Columns):
user_initials_label (categorical).
DebtToIncome (numeric).
final_contract_amount (numeric).
AutomaticFinancing_Score (numeric).
AutomaticFinancing_below_600? (binary/categorical).
AutomaticFinancing_Amount (numeric).
AutomaticFinancing_Details_in_the_wallet? (binary/categorical).
AutomaticFinancing_Details_just_available? (binary/categorical).
AutomaticFinancing_DebtToIncome (numeric).
0UnsecuredFunding_missing? (binary/categorical).
0UnsecuredFunding_Score (numeric).
0UnsecuredFunding_below_600? (binary/categorical).
0UnsecuredFunding_Amount (numeric).
0UnsecuredFunding_PayD (numeric/categorical, unclear without data).
0UnsecuredFunding_Collections (numeric/categorical).
0UnsecuredFunding_DebtToIncome (numeric).
DebtResolution_missing? (binary/categorical).
DebtResolution_Score (numeric).
DebtResolution_below_600? (binary/categorical).
DebtResolution_Amount (numeric).
DebtResolution_DebtToIncome (numeric).
Automatic_Financing_Status_Approved (binary).
Automatic_Financing_Status_Declined (binary).
0%_Unsecured_Funding_Status_As Is (binary).
0%_Unsecured_Funding_Status_Declined (binary).
0%_Unsecured_Funding_Status_If Fixed (binary).
0%_Unsecured_Funding_Status_Missing (binary).
Debt_Resolution_Status_Approved (binary).
Debt_Resolution_Status_Missing (binary).
0UnsecuredFunding_Details_NA (binary).
0UnsecuredFunding_Details_Unsecured Revolving Credit Lines - 0% for 12-21 months (binary).
0UnsecuredFunding_Details_Unsecured Revolving Credit Lines - 0% for 12-21 months - To book: [URL_REMOVED] (binary).

#### NA (Excluded Columns):
record_id, user_initials, final_contract_status, final_contract_tier, final_contract_tier_name, AutomaticFinancing_Status, AutomaticFinancing_Details, 0UnsecuredFunding_Details, 0UnsecuredFunding_Contingencies, DebtResolution_Status, DebtResolution_Details.

# Implementation Methodology
Now, let's strategize how to implement these pre-processing requirements.

## Normalization
Normalization is done in import.py.

## Imputing and Binning/One-hot Encoding
While the import.py json schema is implemented in import.py, the specification of OHE, mapping, and labeling is defined in the JSON file. Other things, though, cannot be implemented in JSON and rely on special code import.py. The import.py special code should be executed prior to the JSON schema fulfillment because it will create certain new values, like NA.

### Done by import.py:
- _below_600 because it requires the score columns to be filled in correctly.
- Both '0% Unsecured Funding' and 'Debt Resolution' offers frequently leave blanks if the offer_status is not 'Approved' for 'Debt Resolution' or 'As Is' for '0% Unsecured Funding'. Any such blanks in 'status' should be made 'Declined' and any blanks in amount should be set to 0. If the score is also blank, it should be imputed a value of -999 and an extra field is added - '_score_missing'. 'Debt Reolution' offers also blank out the 'DebtToIncome' field of 'Declined' offers, so they are also imputed -999. These imputations/overrides all require other information and must be implemented in code.
- Contingencies. '0% Unsecured Funding' has special information in its Contingencies field. This information can be split into 2 columns. The special information is of the form 'PayD-$<value1>' or 'PayD-$<value1>,Collections-$<value2>'
Therefore, we create 2 columns - '0UnsecuredFunding_PayD' and '0UnsecuredFunding_Collections'
0UnsecuredFunding_Contingencies: value1
0UnsecuredFunding_Collections: value2 if value2 is greater than 0; if it is 0, then 0.01.
- Missing offers: Some records don't have all 3 offers. To deal with this, we need to create new columns and impute the data that should be there. The new columns are '0UnsecuredFunding_missing_' and 'DebtResolution_missing_' Where the respective offer is missing, the value in the column should be '1' otherwise '0'. Any missing offers should be filled in with
Name: Debt Resolution | 0% Unsecured Funding
Score: -1
_below_600?: -1
Status: -1
Amount: -1
Details: '' for 'Debt Resolution', NA for '0% Unsecured Funding'
Contingencies: '' for 'Debt Resolution', -1 for '0% Unsecured Funding'
DebtToIncome: -1
- Below 600 scores. For entries that are marked "below 600", we change that to a numeric value of 599 and also add a new field called "_below_600?". All scores below 600 and the ones labeled 'Below 600' get a '1' in this new column. Otherwise, they get a '0'.
Since the scores are associated with the offers, you are adding 3 columns - 'AutomaticFinancing_below_600?', '0UnsecuredFunding_below_600?', and 'DebtResolution_below_600?'

### Done by JSON schema:
- Labels
- Mapping
- One-hot Encoding
  - AutomaticFinancing_Status ->
    - AutomaticFinancing_Status_Approved?
    - AutomaticFinancing_Status_Declined?
  - AutomaticFinancing_Details -->
    - AutomaticFinancing_Details_in_the_wallet?
    - AutomaticFinancing_Details_just_available?
  - 0UnsecuredFunding_Status -->
    - 0UnsecuredFunding_Status_As_is?
    - 0UnsecuredFunding_Status_Declined?
    - 0UnsecuredFunding_Status_if_Fixed?
    - 0UnsecuredFunding_Status_NA?
  - 0UnsecuredFunding_Details -->
    - 0UnsecuredFunding_Details_To_book?
    - 0UnsecuredFunding_Details_just_CL?
    - 0UnsecuredFunding_Details_NA?
  - DebtResolution_Status -->
    - DebtResolution_Status_Approved?
    - DebtResolution_Status_Declined?
    - DebtResolution_Status_NA?
