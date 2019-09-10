#!bash

trap "kill 0" EXIT

# backend app
python app.py &

# give time to load
sleep 15

# frontend app
python hltm_welda/frontend/app.py &

wait
