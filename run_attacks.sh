#!/bin/bash
for atk in 'circular' 'perpendicular';
    do for eps in 1 0.5 0.25 0.1;
        do
        echo $atk $eps
        # cd ./defense/DUP_Net/
        python attack_mesh.py e-mesh --projection $atkatk
        done;done
