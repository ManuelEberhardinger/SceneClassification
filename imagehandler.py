import os
import shutil
from PIL import Image

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# create directory from the classification file for the classification types
# filename: the filename of the classification file
# src: the source folder of the images
# dest: the destination of the src images for the trainingsset
# classification_types: the types as a map, e.g. '0': 'bad_cilia'
# all folders must be in the working directory
def create_directories_for_class(filename, src, dest, classification_types):
    directory = os.getcwd()
    filename = directory + filename
    src = directory + src
    dest = directory + dest
    classification = {}

    # loop through all classifications and save it in a map
    with open(filename, newline='') as csv_file:
        lines = csv.reader(csv_file)
        next(lines)
        for line in lines:
            path = str(line[0].replace('%path%/', src)).strip()
            if ".tif" in path:
                path = path.replace(".tif", ".png")
            classification[path] = str(line[1]).strip()

    # create directories for the classes
    for k, v in classification_types.items():
        create_directory(dest + v)

    not_exists = 0
    moved = 0
    deleted = 0
    already_exists = 0

    # sort images in the right directory for the classes
    for k, v in classification.items():
        if (os.path.exists(k) and v != ''):
            v = classification_types[v]
            try:
                shutil.move(k, dest + v)
                print("File moved: ", k)
                moved = moved+1
            except Exception as e:
                print("File will be deleted: ", k)
                os.remove(k)
                deleted = deleted+1
        elif(v != '' and os.path.exists(dest + classification_types[v]+ '/' + k.split('/')[-1])):
            already_exists = already_exists+1
            continue
        else:
            not_exists = not_exists+1
            print("not exists: ",k,v)

    print("Moved: ", moved, " Deleted: ", deleted, " not exists: ", not_exists, " already moved: ", already_exists)

# changes the image format in png if there are tif images in the trainingset
def change_image_format(folder, format):
    folder = os.getcwd() + folder
    old_img_folder = os.getcwd() + "/tif_images/"
    create_directory(old_img_folder)
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            old_file = os.path.join(root, name)
            if os.path.splitext(old_file)[1].lower() == ".tif":
                if os.path.isfile(os.path.splitext(old_file)[0] + format):
                    print("A " + format + " file already exists for " + name)

                else:
                    outfile = os.path.splitext(old_file)[0] + format
                    try:
                        im = Image.open(old_file)
                        print("Generating " + format + " for " + name)
                        im.thumbnail(im.size)
                        im.save(outfile, "PNG", quality=100)
                        shutil.move(old_file, old_img_folder + name)
                    except Exception as e:
                        print(e)

# augmentation for the images, if the traingssize is to small
# rotate images with the given rotation as an integer
def rotate_images_in_folder(folder, rotation):
    folder = os.getcwd() + folder
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                img = Image.open(path)
                outfile = os.path.splitext(path)[0] + '_' + str(rotation) + ".png"
                print("Rotate Image ", name, " with ", rotation)
                if rotation == 90:
                    img = img.transpose(Image.ROTATE_90)
                elif rotation == 180:
                    img = img.transpose(Image.ROTATE_180)
                elif rotation == 270:
                    img = img.transpose(Image.ROTATE_270)

                img.save(outfile, "PNG", quality=100)

            except Exception as e:
                print(e)
