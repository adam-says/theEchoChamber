# Final unit-suite snapshot

The final frozen candidate suites completed successfully under Python 3.12 and 3.14: **305 passed, 20 subtests** in each run. The copied log files and manifest are byte-preserved snapshots of the run outputs; `snapshot-hashes.json` records their SHA-256 hashes.

The suite ran against uncommitted candidate base `c2e41199191d42b55298baaf46e9c868576d1b2b`. The manifest records hashes for all 74 captured runtime, model, configuration, UI, and test inputs and reports them unchanged through both runs. This preserves the tested-candidate lineage; it does not claim that the candidate had a later commit ID during test execution. Documentation was excluded from the 74-file capture. The neural-mock plotting postprocessor was checked separately and is not included in that source capture. The captured test logs retain their original trailing blank line, so only those `.txt` files are exempt from Git whitespace checks.

From the isolated repository root, reproduce with the corresponding installed interpreter and dependencies:

```powershell
..\venv312\Scripts\python.exe -m pytest -q --junitxml=<fresh-output-dir>\unit-tests-python312.xml
..\venv314\Scripts\python.exe -m pytest -q --junitxml=<fresh-output-dir>\unit-tests-python314.xml
```

The original run outputs, including JUnit XML, are retained outside the repository at `outputs/final-extension-unit-suites-r1/`. The snapshot folder’s `.gitattributes` disables line-ending conversion for these copied evidence files.
