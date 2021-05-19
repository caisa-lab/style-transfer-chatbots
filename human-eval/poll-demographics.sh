#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PIDFILE=${DIR}/poll-demographics.pid

if [ -f $PIDFILE ]
then
  PID=$(cat $PIDFILE)
  ps -p $PID > /dev/null 2>&1
  if [ $? -eq 0 ]
  then
    echo "Process already running"
    exit 1
  else
    ## Process not found assume not running
    echo $$ > $PIDFILE
    if [ $? -ne 0 ]
    then
      echo "Could not create PID file"
      exit 1
    fi
  fi
else
  echo $$ > $PIDFILE
  if [ $? -ne 0 ]
  then
    echo "Could not create PID file"
    exit 1
  fi
fi

cd "$DIR"
python3 poll-demographics-qualification.py

if [[ `git status --porcelain hit-output/demographics*.csv` ]]; then
    # Changes
    git add hit-output/demographics*.csv
    git commit -m "add: new demographics"
    git pull
    git push
fi
now=$(date +"%T")
echo "Current time : $now"

rm $PIDFILE
