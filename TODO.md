- [ ] Documentation:
  - [ ] Result tracker: a report from the intensive runs with the best candidate's full metrics + feature importances. It should identify what changed in the source vs. what the resulting top candidate's results were. It needs to be maintained with every major improvement, so add it to the end of every Todo list.Ideally it has a graph showing f1 score of the top candidate on the y-axis and date on the x-axis.
  - [ ] SRS
  - [ ] validation plan
  - [ ] Developer Guide: How the code and execution is organized. How can a new person jump into making a positive contribution? What are the current challenges?
  - [ ] Architecture document containing:
    - [ ] high-level overview
    - [ ] function-by-function crawl
    - [ ] current challenges
    - [ ] future enhancement opportunities
  - [ ] Review: a line-by-line explanation of what the code does, why it is necessary, and what enhancements are needed. This is a huge file.
- [ ] Pylance error cleanup
- [ ] Maintain `docs/RESULTS_TRACKER.md` with every major improvement (append a new report and update `docs/results_history.csv`)

---

--dryrun is okay
--50 samples/classifier had issues, especially with SMOTE changing from int to float.

---

- [x] Enforce int->float conversion integrity: changing 3.0 (float) to int(3) is allowed; introducing fractional floats (e.g., 3.1) is treated as an error and will fail validation.
- [ ] Clean up output, especially progress bar and ETA.
  - [ ] What, if anything, needs done to the high-correlation errors. I presume it means that we don't need both features and should drop at least 1 of them. That should also come out in the feature importance rankings. Does it?
- [ ] Are you monitoring the cv frames to see how the f1 score is trending? What do you see, in terms of our 90% goal?
- [ ] How does the final classifier perform on the test data? Update documentation.


Here's the results of my QA:
- user_1254_credit_summary_2025-09-01_095528.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1254_credit_summary_2025-09-01_095528.pdf,$1159 Unpaid Collection(s),#ebeff4,neutral,ocr ### This is green instead of neutral
- user_1254_credit_summary_2025-09-01_095528.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1254_credit_summary_2025-09-01_095528.pdf,2 RE Lates in 0-3 mo,#ebeff4,neutral,ocr ### This is green instead of neutral
- There are many more that are missing from user_1254_credit_summary_2025-09-01_095528.pdf
    - 3 Over Limit Accnt (green)
    - Current Lates (CREDIT ACCEPTANCE CO) [green]
    - Past Due Not Late [green]
    - 4 Chrgd Off Rev Accts [red]
    - 1 Rev Late in 0-3 mo [red]
    - Avg Age Open [red]
    - No 5k+ Lines [red]
    - No Closed Rev Depth [red]
    - 3 RE Lates in 2-4 yrs [red]
    - 1 Rev Late in 2-4 yrs [red]
    - 2 inqs last 2 Months [red]
    - 2 Total Inq 0-2 Mo [red]
    - Ok Open Rev Depth [green]
    - 3+ Closed Rev Accnts [green]
    - $440 Unpaid 1 Collection [black]
    - No Open Mortgage [black]
    - No Rev Acct Open 10K 2yr [black]
    - 1 inq last 2-4 Mo [black]
    - 1 Total Inq 2-5 Mo [black]
- These should have been green instead of neutral:
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,3 Charged Off Accts,#919395,neutral,ocr
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,4 Unpaid Collection(s),#ebeff4,neutral,ocr
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,$42683 Unpaid Collection(s),#ebeff4,neutral,ocr
- Many missing factors for user_1514_credit_summary_2025-09-01_145557.pdf:
    - 7 Over Limit Accts [green]
    - Total Rev Usage > 90 [green]
    - Past Due Not Late [green]
    - Pay $12355 so Accts < 40 [red]
    - 8 Chrgd Off Rev Accts [red]
    - 2 Rev Lates in 4-6 mo [red]
    - 2 Rev Lates in 6-12 mo [red]
    - $3029 Unpaid Collections [red]
    - 4 Rev Lates in 2-4 yrs [red]
    - 1 RE Late in 4-6 mo [red]
    - No 5k+ Lines [red]
    - 1 Chrgd Off RE Acct [red]
    - 1 Inq Last 4 Mo [red]
    - Less than 5 yrs [red]
    - Great Closed Rev Depth [green]
    - Ok Open Rev Depth [green]
    - 3+ Closed Rev Accts [green]
    - Military Affiliated [green]
    - Seasoned Closed Accounts [green]
    - Closed Accnts Over 5k [green]
    - 8+ Rev Accnts with Balances [black]
    - In Credit Counseling [black]
    - No Open Mortgage [black]
    - 1 Rev Late in 4+ yrs [black]
    - Drop Bad Auth User (BEST BUY/CBNA, and C [black]
    - No Rev Acct Open 10K 2yr [black]
    - 1 Inq last 2-4 Mo [black]
    - 1 Inq Last 4-5 mo [black]
    - 1 Total Inq 2-4 Mo [black]

In credit_factors_poc_wide.csv:
- The credit scores are correct
- The ages are correct.
- has collections wasn't one of the requested columns, but maybe it should have been. True is correct, but we can go the next level of detail. The specific counts are:
    - 12 open and 10 closed.
    - 20 open and 1 closed.
    -  3 open and 2 closed.
- The counts are off. They should be:
    - 11 red; amber isn't a color on the list; 14 greens; 9 blacks.
    - 13 red; amber isn't a color on the list; 17 greens; 4 blacks.
    - 12 red; amber isn't a color on the list; 10 greens; 5 blacks.

For poc_images, 2 of the credit factors spilled onto a second page but you only got page 1.

credit_factors_poc_qa.json with parsed color counts summary is incorrect. The correct values are:
    - 11 red; neutral isn't a color the document; 14 greens; 9 blacks.
    - 13 red; neutral isn't a color in the document; 17 greens; 4 blacks.
    - 12 red; neutral isn't a color in the document; 10 greens; 5 blacks.

