import os

def find_image(img_dir):
    filenames = os.listdir(img_dir)
    for i, filename in enumerate(filenames):
        if not filename.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            filenames.pop(i)
    
    return filenames

def generate_filelist(img_dir, valid=False):
    # get filenames list
    filenames = find_image(img_dir)
    if len(filenames) == 0:
        filenames = find_image(os.path.join(img_dir, 'input'))
        if len(filenames) == 0:
            raise(f"No image in directory: '{img_dir}' or '{os.path.join(img_dir, 'input')}'")

    # write filenames
    filelist_name = 'val_list.txt' if valid else 'train_list.txt'
    with open(os.path.join(filelist_name), 'w') as f:
        for filename in filenames:
            print(filename)
            f.write(filename + '\n')

if __name__ == '__main__':
    generate_filelist('../dataset/test_samples/trainA/')