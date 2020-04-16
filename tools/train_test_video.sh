#!/bin/bash
python tools/train.py $* --seed 0 --validate
echo ""
echo "Runing test script:"
python tools/test_video.py $*
