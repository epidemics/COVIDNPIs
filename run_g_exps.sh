#!/bin/bash
#!/bin/bash
for g in 0.01 0.02 0.03 0.05 0.1 0.15 0.2 0.3 0.4
do
     echo "running with noise $g"
     python run_g_exp.py --g $g --s 2000 --c 4 --rg &
done