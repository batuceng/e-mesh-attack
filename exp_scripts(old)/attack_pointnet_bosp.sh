python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection perpendicular --alpha 0.01 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection perpendicular --alpha 0.02 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection perpendicular --alpha 0.05 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection perpendicular --alpha 0.10 --steps 250

python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection central --alpha 0.01 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection central --alpha 0.02 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection central --alpha 0.05 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  e-mesh --projection central --alpha 0.10 --steps 250

python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  pgd --eps 0.01 --alpha 0.0001 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  pgd --eps 0.05 --alpha 0.002 --steps 250

python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  pgdl2 --eps 1.25 --alpha 0.01 --steps 250
python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  pgdl2 --eps 1.25 --alpha 0.05 --steps 250

python attack_mesh.py --dataset bosphorus --z-filter False -model pointnet  vanila