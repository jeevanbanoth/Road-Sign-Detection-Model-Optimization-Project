
#!/bin/bash
# Loop from 1 to 30
for i in {1..30}
do
   echo "Running Parallel.py with $i threads"
   python3 Parallel.py $i
done
