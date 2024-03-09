# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-3 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-4 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 4e-3 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 8e-3 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-3 --wd 1e-4 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-3 --wd 4e-5 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 4e-3 --wd 4e-5 --lr_step_size 40 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-3 --wd 1e-6 --lr_step_size 25 --lr_decay 0.95
# python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-1 --wd 1e-5 --lr_step_size 15 --lr_decay 0.95

python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 1e-4 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 2e-4 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 4e-4 --wd 1e-5 --lr_step_size 25 --lr_decay 0.95
python train_classifier.py --dataset bosphorus --model pointnet --z-filter False --lr 2e-4 --wd 1e-6 --lr_step_size 25 --lr_decay 0.95

# zip -r myarchive.zip dir1 -x dir1/ignoreDir1/**\* dir1/ignoreDir2/**\*
