echo "application,successful" > dlmt_valid_runtime.csv && find */*  | grep -v "random" | grep "log" | xargs grep -n "Stored as a" | cut -d: -f1,3 | sed "s/[/: ]/,/g" | cut -d, -f1,7 | sed "s/sucessful/\"True\"/g" | sed "s/failed/\"False\"/g" >> dlmt_valid_runtime.csv

echo "application,successful" > rs_valid_runtime.csv && find */*  | grep "random" | grep "log" | xargs grep -n "Stored as a" | cut -d: -f1,3 | sed "s/[/: ]/,/g" | cut -d, -f1,7 | sed "s/sucessful/\"True\"/g" | sed "s/failed/\"False\"/g" >> rs_valid_runtime.csv
