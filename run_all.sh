#!/bin/bash
for i in {1..7}
do
  echo "***  starting exp"$i
  python exp$i.py
done
