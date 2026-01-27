---
agent: agent
---
Rules:
1. Do not use token matching to identify color. The extraction must be generic and work for thousands of cases.
2. Do not use environment variables for flow control. Environment variables should only be used for secrets. Flow control variables should be set in the config/control.ini file.
3. Before delivering code, ensure it adheres to PEP 8 standards for Python code.
4. Write unit tests for any new functionality added. Ensure that all existing and new unit tests pass before delivering code.
    a) Use pytest for writing and running tests.
    b) Aim for at least 80% code coverage on new modules on the first delivery.
    c) Include tests for edge cases and error handling.
    d) Document the tests clearly, explaining what each test is verifying.
    e) Organize tests in a separate `tests/` directory, mirroring the structure of the `src/` directory.
5. When updating documentation, ensure that all relevant sections are updated to reflect the changes.
6. In addition to unit tests, write integration tests for any new modules that interact with existing components.
    a) Ensure that integration tests cover the interaction between new and existing modules.
    b) Use realistic data scenarios in integration tests to simulate real-world usage.
    c) Document integration tests clearly, explaining the purpose and expected outcomes.
    d) Starting integration tests are tests/test_pdf_extraction_ground_truth.py run with cli arguments for users whose ground truths are validated in data/extracted, as:
    - `pytest tests/test_pdf_extraction_ground_truth.py --user_id 582` because data/extracted/user_582_credit_summary_*ground_truth.json exists (validated grround truth file)