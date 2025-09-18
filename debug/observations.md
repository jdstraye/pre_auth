For final_contract_status
    "labels":{
      "approved": "0",
      "declined": "1",
      "rejected": "0",
      "refunded": "0"
    }
#============================
[20250910.1206] logs/
384 Sep 10 12:00 status_test_report.csv
SUMMARY - The model is mostly just guessing approved.
STATUS Accuracy: 81%
'Approved' Precision: 82%
'Approved' Recall: 96%
'Approved' F1: 88.8%
'Declined' Precision: 70%
'Declined' Recall: 30%
'Declined' F1: 42%

,precision,recall,f1-score,support
0,0.8260869565217391,0.9620253164556962,0.8888888888888888,79.0
1,0.7,0.30434782608695654,0.42424242424242425,23.0
accuracy,0.8137254901960784,0.8137254901960784,0.8137254901960784,0.8137254901960784
macro avg,0.7630434782608695,0.6331865712713264,0.6565656565656566,102.0
weighted avg,0.7976555839727194,0.8137254901960784,0.7841156664686076,102.0

Details of logs/status_test_preds.csv

There are 16 "declined" that are predicted as "approved"; 7 true "declined" are predicted as "declined"
There are 3 "approved" that are predicted as "declined"; 76 true "approved" are predicted as "approved"

Details of logs/pre_auth_eval_algos-20250910.1155.log

Feature importance - Not captured
Features used - "Initializing NamedSMOTENC with categorical features" is always the same- ['AutomaticFinancing_below_600_', 'AutomaticFinancing_Details_in_the_wallet_', 'AutomaticFinancing_Details_just_available_', 'AutomaticFinancing_Details_NA_', '0UnsecuredFunding_missing_', '0UnsecuredFunding_below_600_', '0UnsecuredFunding_Details_To_book_', '0UnsecuredFunding_Details_NA_', 'DebtResolution_missing_', 'DebtResolution_score_missing_', 'DebtResolution_below_600_']

#============================
[20250910.1512] logs/
5379 Sep 10 15:11 pre_auth_eval_algos-20250910.1511.log
Feature Importances are working :)
Status top features -
final_contract_amount, DebtToIncome, AutomaticFinancing_DebtToIncome, 0UnsecuredFunding_status_Declined
Tier top features -
AutomaticFinancing_Score, DebtResolution_Score, 0UnsecuredFunding_Score, DebtToIncome

Useless features - These features Feature Importance score never went above 0:
CMDs =
grep -En "FI:" logs/pre_auth_eval_algos_latest.log | awk '{print $NF, $0}' | sort -n -r | cut -d ' ' -f 2- > /tmp/foo
awk -F' - FI: ' '{split($2,a,": "); arr[a[1]] = (a[2] > arr[a[1]] ? a[2] : arr[a[1]])} END {for (i in arr) { if (arr[i] == 0) print i}}' /tmp/foo

AutomaticFinancing_Details_NA_, AutomaticFinancing_Status_NA_

Different classifiers give different scores, so it can be hard to compare after that, but here's the ranges seen in this run -
Higher Value features:
AutomaticFinancing_DebtToIncome: Highest: 2629.0000, Lowest: 6.5518
DebtToIncome: Highest: 1658.0000, Lowest: 5.8357
AutomaticFinancing_Score: Highest: 1615.0000, Lowest: 10.2111
AutomaticFinancing_Amount: Highest: 1528.0000, Lowest: 5.5455
DebtResolution_Score: Highest: 1386.0000, Lowest: 6.7358
0UnsecuredFunding_Score: Highest: 1159.0000, Lowest: 6.6092
final_contract_amount: Highest: 1011.0000, Lowest: 23.4555
0UnsecuredFunding_Amount: Highest: 978.0000, Lowest: 5.5885
DebtResolution_DebtToIncome: Highest: 1117.0000, Lowest: 6.8638

Medium Value Features:
0UnsecuredFunding_PayD: Highest: 780.0000, Lowest: 8.0000
DebtResolution_Amount: Highest: 622.0000, Lowest: 7.0120
AutomaticFinancing_below_600_: Highest: 378.0000, Lowest: 6.1297
DebtResolution_below_600_: Highest: 301.0000, Lowest: 6.8638
0UnsecuredFunding_DebtToIncome: Highest: 235.0000, Lowest: 6.0577
AutomaticFinancing_Status_Declined: Highest: 209.0000, Lowest: 9.0000

Low Value Features:
AutomaticFinancing_Status_Approved: Highest: 145.0000, Lowest: 5.8355
DebtResolution_score_missing_: Highest: 85.0000, Lowest: 6.0000
0UnsecuredFunding_status_Declined: Highest: 70.0000, Lowest: 7.0819
0UnsecuredFunding_status_If_Fixed: Highest: 63.0000, Lowest: 8.0000
0UnsecuredFunding_status_As_Is: Highest: 43.0000, Lowest: 14.0000
DebtResolution_missing_: Highest: 30.0000, Lowest: 7.0000
0UnsecuredFunding_below_600_: Highest: 19.9620, Lowest: 7.3538
AutomaticFinancing_Details_just_available_: Highest: 18.0000, Lowest: 18.0000
AutomaticFinancing_Details_in_the_wallet_: Highest: 10.5694, Lowest: 0.0000
0UnsecuredFunding_Details_To_book_: Highest: 8.0000, Lowest: 0.0000

********
********
****************
********
********

For final_contract_status
For final_contract_status
    "labels":{
      "approved": "0",
      "declined": "1",
      "rejected": "2",
      "refunded": "0"
    }
#============================
[20250910.1206] logs/
1767 Sep 10 13:49 pre_auth_eval_algos-20250910.1349.log
SUMMARY - The model is mostly just guessing approved; it completely ignores 'rejected'
STATUS Accuracy: 70%
'Approved' Precision: 72%
'Approved' Recall: 98%
'Approved' F1: 83%
'Declined' Precision: 55%
'Declined' Recall: 22%
'Declined' F1: 31%
'Rejected' Precision: 0%
'Rejected' Recall: 0%
'Rejected' F1: 0%

,precision,recall,f1-score,support
0,0.7204301075268817,0.9852941176470589,0.8322981366459627,68.0
1,0.5555555555555556,0.21739130434782608,0.3125,23.0
2,0.0,0.0,0.0,11.0
accuracy,0.7058823529411765,0.7058823529411765,0.7058823529411765,0.7058823529411765
macro avg,0.42532855436081246,0.4008951406649617,0.3815993788819876,102.0
weighted avg,0.6055590695059386,0.7058823529411765,0.6253311107051517,102.0
CONCLUSION: This is clearly wrong.