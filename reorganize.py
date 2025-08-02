import os
import shutil

original_train_dir = '/Users/pk/Projects/x-ray-image-analysis/MURA-v1.1/train'
new_train_dir = 'data/train'

folder_map = {
    'XR_ELBOW': 'elbow',
    'XR_HAND': 'hand',
    'XR_SHOULDER': 'shoulder',
    'XR_WRIST': 'wrist',
    'XR_FINGER': 'finger',
    'XR_FOREARM': 'forearm',
    'XR_HUMERUS': 'humerus',
}

os.makedirs(new_train_dir, exist_ok=True)

max_images_per_class = 200

total_copied = 0
for mura_folder, simple_label in folder_map.items():
    src_folder = os.path.join(original_train_dir, mura_folder)
    tgt_folder = os.path.join(new_train_dir, simple_label)
    os.makedirs(tgt_folder, exist_ok=True)

    print(f'\nProcessing folder: {mura_folder} → {simple_label}')
    patients = os.listdir(src_folder)
    print(f'Found {len(patients)} patient folders in {mura_folder}')

    copied_count = 0
    for patient_folder in patients:
        if copied_count >= max_images_per_class:
            break
        patient_path = os.path.join(src_folder, patient_folder)
        if not os.path.isdir(patient_path):
            print(f'Skipping non-folder: {patient_path}')
            continue

        study_folders = os.listdir(patient_path)
        for study_folder in study_folders:
            if copied_count >= max_images_per_class:
                break
            study_path = os.path.join(patient_path, study_folder)
            if not os.path.isdir(study_path):
                continue

            images = os.listdir(study_path)
            # Filter image files, skip mac hidden files
            images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg')) and not img.startswith('._')]
            print(f'  {patient_folder}/{study_folder} has {len(images)} images')

            for image_name in images:
                if copied_count >= max_images_per_class:
                    break
                src_img_path = os.path.join(study_path, image_name)
                dst_img_name = f"{patient_folder}_{study_folder}_{image_name}"
                dst_img_path = os.path.join(tgt_folder, dst_img_name)
                shutil.copy2(src_img_path, dst_img_path)
                copied_count += 1
                total_copied += 1
    print(f'Copied {copied_count} images for {simple_label}')

print(f'\nDataset subset creation complete! Total images copied: {total_copied}')
