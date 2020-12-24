#!/usr/bin/python

import optuna
import sys

study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///" + sys.argv[1])

print(study_summaries[0].n_trials)
print(study_summaries[0].best_trial)

