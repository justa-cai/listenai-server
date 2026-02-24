rm -fv tmp/*
python3 asr_websocket_server.py 2>&1 | tee -a log.txt