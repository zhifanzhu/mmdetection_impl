#!/bin/bash
python tools/train.py $*
python tools/test_video.py $*
