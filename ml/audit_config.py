"""Round-2 settings. Decide these BEFORE round 2 starts and write them in
Section 3.6.4; do not edit them after seeing round-2 results.

SUPERSEDED_ROUND1: annotators whose round-1 labels are replaced by round-2
labels because their agreement with the other annotators was far below the
rest (audit_diagnostics.py, Section 1). All six annotators are still
reported in the round-1 results.
"""
SUPERSEDED_ROUND1 = ["E", "F"]
REDO_ADJ_PAIR = "EF"            # this pair's adjudication file is redone
REDO_ADJ_TO = ["AB", "CD"]      # ...by these pairs, half each
FLEISS_TARGET = 0.60            # round-2 shared set, six annotators
BINARY_KAPPA_MIN = 0.70         # same threshold as round 1
