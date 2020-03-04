#!/bin/bash
python tools/train.py $* --validate
echo ""
echo "Runing test script:"
python tools/test_video.py $*
