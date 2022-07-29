#!/bin/bash
set -e
cd /work/ihaide/GirlsDay
echo 'Working in the folder:'; pwd
echo 'Setting up the environment'
echo 'Current environment:'; env
echo 'Will now execute the program'
exec /work/ihaide/miniconda3/envs/metrics/bin/python ntuples.py --batch-runner --task-id ParticleGunWrapperTask__99914b932b