from torchvision import datasets


class MyImageFolder(datasets.ImageFolder):
    # Overwrite getitem so you get also a path for the image file.
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]


class ImageSplitter(datasets.ImageFolder):

    # split image in 4 parts, so a 224x224 image will be four 112x112 images
    def split(self):
        splitted_imgs = []
        count = len(self.imgs)
        for path, target in self.imgs:
            img = self.loader(path)
            width, height = img.size

            middle_width = width / 2
            middle_height = height / 2

            bounds = (0, 0, middle_width, middle_height)
            splitted_imgs.append((img.crop(bounds), target))
            bounds = (middle_width, 0, width, middle_height)
            splitted_imgs.append((img.crop(bounds), target))
            bounds = (0, middle_height, middle_width, height)
            splitted_imgs.append((img.crop(bounds), target))
            bounds = (middle_width, middle_height, width, height)
            splitted_imgs.append((img.crop(bounds), target))

        self.imgs.clear()
        self.imgs = splitted_imgs
        print("Images before: " + str(count) + " Now: " + str(len(self.imgs)))

    # Overwrite get item of pytorch dataloader because self.imgs variable is of a different type now
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

