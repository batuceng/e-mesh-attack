#!/bin/bash

for atk in 'vanila';
    do for model in 'dgcnn' 'pointnet';
        do
        echo $model $atk
        python attack_mesh.py -model $model $atk 
done;done

for atk in 'e-mesh';
    do for model in 'dgcnn' 'pointnet';
        do for proj in 'central' 'perpendicular';
            do for eps in 1 0.5 0.25 0.1;
                do for steps in 1 5 10 25 50;
                    do
                    echo $model $atk $eps $proj $steps
                    python attack_mesh.py -model $model $atk --eps $eps --projection $proj --steps $steps
done;done;done;done;done;

for atk in 'pgdl2';
    do for model in 'dgcnn' 'pointnet';
        do 
        for eps in 1.25 0.625 0.3125 0.15625;
            do
            echo $model $atk $eps
            python attack_mesh.py -model $model $atk --eps $eps
        done;
        for steps in 1 5 10 25 50;
            do
            echo $model $atk $steps
            python attack_mesh.py -model $model $atk --steps $steps
        done;
done;done;

for atk in 'pgd';
    do for model in 'dgcnn' 'pointnet';
        do 
        for eps in 0.05 0.025 0.0125;
            do
            echo $model $atk $eps
            python attack_mesh.py -model $model $atk  --eps $eps
        for steps in 1 5 10 25 50;
            do
            echo $model $atk $steps
            python attack_mesh.py -model $model $atk  --steps $steps
done;done;done
