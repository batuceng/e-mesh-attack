import os
from stl import mesh
from pathlib import Path

if __name__ == "__main__":
    
    # Path to folder of Facewarehouse dataset (stl files)
    main_folder = 'FaceWarehouse_stl'
    subfolder_list = os.listdir(main_folder)

    # Rename files to fit to the dataloader
    for tester in subfolder_list:
        current_path = os.path.join(main_folder,tester)
        print(current_path)
        tester_no = int(tester.split('_')[-1])
        tester_word = tester.split('_')[0]
        os.rename(os.path.join(main_folder,tester),
                os.path.join(main_folder, f'{tester_word}_{tester_no:03}'))