set -x
lsof -ti:9401 | xargs -r kill -9
python3 -m src 2>&1 | tee log.txt
