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
- 'Debt Resolution_score_missing?'
- '0% Unsecured Funding_Contingencies'
- '0% Unsecured Funding_Collections'
- 'Automatic Financing_below_600?'
- '0% Unsecured Funding_below_600?'
- 'Debt Resolution_below_600?'
- '0% Unsecured Funding_missing?'

## Deleted Columns/Fields
Other fields in the JSON did not have any distinguishing data and were removed:
- 'Debt Resolution_Details'
- 'Debt Resolution_Contingencies'
- 'Automatic Financing_Contingencies'

## Offer anomalies
### Empty Data for Rejections
Both '0% Unsecured Funding' and 'Debt Resolution' offers frequently leave blanks if the offer_status is not 'Approved' for 'Debt Resolution' or 'As Is' for '0% Unsecured Funding'. Any such blanks in 'status' should be made 'Declined' and any blanks in amount should be set to $0

### Missing Offers
Some records don't have all 3 offers. To deal with this, we need to create new columns and impute the data that should be there. The new columns are '0% Unsecured Funding_missing?' and 'Debt Resolution_missing?' Where the respective offer is missing, the value in the column should be '1' otherwise '0'. Any missing offers should be filled in with
Name: Debt Resolution | 0% Unsecured Funding
Score: -1
_below_600?: -1
Status: NA
Amount: -1
Details: NA
Contingencies: 'NA' for 'Debt Resolution', -1 for '0% Unsecured Funding'
DebtToIncome: -1

### Contingencies
'0% Unsecured Funding' has special information in its Contingencies field. This information can be split into 2 columns. The special information is of the form 'PayD-$<value1>' or 'PayD-$<value1>,Collections-$<value2>'
Therefore, we create 2 columns - '0% Unsecured Funding_Contingencies' and '0% Unsecured Funding_Collections'
0% Unsecured Funding_Contingencies: value1
0% Unsecured Funding_Collections: value2 if value2 is greater than 0; if it is 0, then 0.01.

## Numerical Columns with Buckets
### Below 600 scores
For entries that are marked "below 600", we change that to a numeric value of 599 and also add a new field called "_below_600?". All scores below 600 and the ones labeled 'Below 600' get a '1' in this new column. Otherwise, they get a '0'.
Since the scores are associated with the offers, you are adding 3 columns - 'Automatic Financing_below_600?', '0% Unsecured Funding_below_600?', and 'Debt Resolution_below_600?'

## Missing Not At Random (MNAR)
### Debt Resolution Offers
Debt Resolution hides the score if the offer is Declined but provides it if Approved. For these cases, fill in the score with -999 and create a new column - 'Debt Resolution_score_missing?' and populate it with '1' if the score is missing, '0' otherwise.

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
For input features with a low number of distinct categories, one-hot encoding was used to avoid  implying a false ordinal relationship. Specifically, it was applied to 
- Automatic Financing_Status ->
  - Automatic Financing_Status_Approved?
  - Automatic Financing_Status_Declined?
- Automatic Financing_Details -->
  - Automatic Financing_Details_in_the_wallet?
  - Automatic Financing_Details_just_available?
- 0% Unsecured Funding_Status -->
  - Unsecured Funding_Status_As_is?
  - Unsecured Funding_Status_Declined?
  - Unsecured Funding_Status_if_Fixed?
  - Unsecured Funding_Status_NA?
- 0% Unsecured Funding_Details -->
  - 0% Unsecured Funding_Details_To_book?
  - 0% Unsecured Funding_Details_just_CL?
  - 0% Unsecured Funding_Details_NA?
- Debt Resolution_Status -->
  - Debt Resolution_Status_Approved?
  - Debt Resolution_Status_Declined?
  - Debt Resolution_Status_NA?

## Output
If the result were implemented in a table, it would have the following columns:
Column Name	Data Type	Description
record_id	string	The unique identifier for each loan application.
user_initials	string	The user's initials, should be one-hot encoded for the model.
DebtToIncome	float	The applicant's debt-to-income ratio.
final_contract_status	integer	The target for the first model. Will be label encoded (e.g., declined=0, approved=1).
final_contract_tier	integer	The target for the second model. Will be label encoded (0-9).
Automatic Financing_Name	string	The name of the offer. Will be one-hot encoded.
Automatic Financing_Score	float	The applicant's score, with "below 600" imputed as 599.
Automatic Financing_below_600?	integer	A binary flag (1 or 0) indicating if the score is <600.
Automatic Financing_Status	string	The status of the offer. Will be one-hot encoded.
Automatic Financing_Amount	float	The offer amount in dollars.
Automatic Financing_Details	string	Offer details, will be one-hot encoded.
0% Unsecured Funding_Name	string	Name of the offer.
0% Unsecured Funding_Score	float	The offer's score, with "below 600" imputed as 599.
0% Unsecured Funding_below_600?	integer	A binary flag (1 or 0) for scores <600.
0% Unsecured Funding_Status	string	Status of the offer. "" (empty) has been imputed to "Declined".
0% Unsecured Funding_Amount	float	The offer amount. "" (empty) has been imputed to 0.
0% Unsecured Funding_Details	string	Details of the offer.
0% Unsecured Funding_Contingencies	float	The numerical PayD value, extracted from the string.
0% Unsecured Funding_Collections	float	The numerical Collections value, extracted from the string.
0% Unsecured Funding_DebtToIncome	float	The DebtToIncome for this offer.
0% Unsecured Funding_missing?	integer	A binary flag (1 or 0) for records with a completely missing offer.
Debt Resolution_Name	string	Name of the offer.
Debt Resolution_Score	float	Score, with declined offers imputed as -999.
Debt Resolution_below_600?	integer	A binary flag (1 or 0) for scores <600.
Debt Resolution_score_missing?	integer	A binary flag (1 or 0) for declined offers with a missing score.
Debt Resolution_Status	string	Status of the offer. "" (empty) has been imputed to "Declined".
Debt Resolution_Amount	float	The offer amount. "" (empty) has been imputed to 0.
Debt Resolution_Details	string	Details of the offer.
Debt Resolution_Contingencies	string	Contingencies of the offer. "" (empty) has been imputed to "NA".
Debt Resolution_DebtToIncome	float	The DebtToIncome for this offer.
Debt Resolution_missing?	integer	A binary flag (1 or 0) for records with a completely missing offer.