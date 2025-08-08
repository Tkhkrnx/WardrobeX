# data/extract_fashiongen_data.py
import os
import csv
import h5py
from PIL import Image

headline = ['index', 'description', 'category']

def extract_data(h5_path, csv_output, img_output_dir):
    os.makedirs(img_output_dir, exist_ok=True)
    file_h5 = h5py.File(h5_path, 'r')

    csvfile = open(csv_output, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=headline)
    writer.writeheader()

    for i in range(len(file_h5['index'])):
        index = int(file_h5['index'][i][0])
        try:
            category = str(file_h5['input_category'][i][0], 'UTF-8')
            description = str(file_h5['input_description'][i][0], 'UTF-8')
        except:
            continue

        img = Image.fromarray(file_h5['input_image'][i])
        image_path = os.path.join(img_output_dir, f'{index}.jpg')
        img.save(image_path)

        writer.writerow({'index': str(index), 'description': description, 'category': category})

    csvfile.close()
    file_h5.close()
    print(f"Extracted {h5_path} to {img_output_dir} and {csv_output}")


if __name__ == '__main__':
    # 确保h5文件存在再运行
    if os.path.exists('fashiongen_256_256_train.h5'):
        extract_data('fashiongen_256_256_train.h5', 'fashiongen_train.csv', './images/train')
    if os.path.exists('fashiongen_256_256_validation.h5'):
        extract_data('fashiongen_256_256_validation.h5', 'fashiongen_val.csv', './images/val')
