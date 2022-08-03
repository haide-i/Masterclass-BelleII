#!/bin/bash

#sourcing basf
source /cvmfs/belle.cern.ch/tools/b2setup
b2setup release-06-00-03

export IDMDH_PATH=/work/jeppelt/Anomaly_Detection_IDM/idmwithdh
export HOME=/home/jeppelt
export _CONDOR_SCRATCH_DIR=/ceph/jeppelt/temp

export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.8/site-packages/
export PATH=$PATH:$HOME/.local/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/belle.cern.ch/el7/externals/v01-10-02/Linux_x86_64/common/lib/

#source /cvmfs/belle.kek.jp/grid/gbasf2/pro/tools/setup.sh
