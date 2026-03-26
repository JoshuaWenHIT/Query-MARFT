#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat configs/motrv2.args)
python3 submit_dance.py ${args} --exp_name tracker --resume $1
