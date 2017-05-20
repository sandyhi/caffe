#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/rank_net/rank_net_solver.prototxt $@
