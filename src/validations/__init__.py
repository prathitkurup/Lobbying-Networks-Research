# Validation scripts for the Fortune 500 lobbying network project.
# Each script is self-contained and can be run from the src/ directory:
#   python validations/01_extraction_audit.py
#
# Run order:
#   01 → extraction audit (understand raw data structure)
#   02 → inflation diagnosis (quantify and fix cartesian product bug)
#   03 → sparsity analysis (null model, 27x above expectation finding)
#   04 → mega_bill diagnosis (prevalence filtering rationale)
#
# Outputs are written to validations/outputs/ as human-readable .txt files.
