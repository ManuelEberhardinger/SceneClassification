from torchvision import transforms
import customtransforms
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import os
import json


class TransformationHelper:

    def __init__(self):
        self.std_image_size = 224
        self.std_scale = 256

    def get_trainings_transformations(self):
        return transforms.Compose([
                #transforms.Resize(self.std_scale),
                #transforms.Lambda(lambda x: x.convert('L')),
                transforms.RandomResizedCrop(self.std_image_size),
                transforms.RandomHorizontalFlip(),
                customtransforms.RandomVerticalFlip(),
                customtransforms.RandomRotation(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_test_transformations(self):
        return transforms.Compose([
                #transforms.Lambda(lambda x: x.convert('L')),
                transforms.Resize(self.std_scale),
                transforms.CenterCrop(self.std_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # for doublets and small images
    def change_std_img_and_scale(self):
        self.std_image_size =  112
        self.std_scale = 128

    # for the normal images and the predefined models
    def reset_to_std_img_and_scale(self):
        self.std_image_size = 224
        self.std_scale = 256


class TestHelper:

    def __init__(self, data_loader, dset_classes, net, device):
        self.data_loader = data_loader
        self.classes = dset_classes
        self.size = len(dset_classes)
        self.net = net.to(device)
        self.class_correct = list(0. for i in range(self.size))
        self.class_total = list(0. for i in range(self.size))
        self.device = device

    # test the total precision for two classes
    def test_total_precision(self):
        total = correct = 0
        for data in self.data_loader:
            (images, labels_data), (path, _) = data
            labels = (Variable(labels_data)).to(self.device)
            outputs = self.net((Variable(images)).to(self.device))
            probs = F.softmax(outputs, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            c = (predicted == labels).squeeze()
            label = labels[0]
            self.class_correct[label] += c.item()
            self.class_total[label] += 1
            probs = probs.data.cpu().numpy()[0]
            percentage_probs = [i * 100 for i in probs]
            predicted = predicted.cpu().numpy()[0]
            print(path[0], "Predicted: ", self.classes[predicted])
            print(self.classes)
            print(percentage_probs)
            print()
            labels = labels.cpu().numpy()[0]
            #if (predicted != labels):
                # print the false classified test data
                #print(path[0] + ': ' + self.classes[predicted] + " " + str(probs[0]) + " " + str(probs[1]))
                #shutil.copy2(path[0], false_pred + path[0].split("/")[-1][:-4] + "_" + self.dset_classes[predicted] + ".png")

    # print the statistics of the classes, prints also the sensitivity and specificity
    def print_total_precision(self, name, epoch):
        with open('results.txt', 'a') as f:
            print(name, "Epoch:", epoch, file=f)
            for i in range(self.size):
                if(self.class_total[i] == 0):
                    continue
                print('Accuracy of %5s : %2d %%' % (
                    self.classes[i], 100 * self.class_correct[i] / self.class_total[i]), file=f)
                print(self.classes[i] + ": " + str(self.class_correct[i]) + " of " + str(self.class_total[i]) + " images", file=f)

            print("Total images: " + str(len(self.data_loader)), file=f)
            correct = 0
            for i in range(len(self.class_correct)):
                correct = correct + self.class_correct[i]

            print("Total precision: %2d %%" % (100*correct / sum(self.class_total)), file=f)


class JsonLocationLoader:

    def __init__(self, path, classes):
        self.path = path
        self.classes = classes
        self.dict = {}

    def load_all_locations(self):
        for label in self.classes:
            self.dict[label] = self.load_locations_for_label(label)
        return self.dict

    def load_locations_for_label(self, label):
        label_path = os.path.join(self.path, label)
        locations = {}
        files = [x for x in os.listdir(label_path) if x.endswith("json")]
        for file in files:
            json_file_path = os.path.join(label_path, file)
            with open(json_file_path, "r") as f:
                image_id = os.path.splitext(file)[0]
                content = json.load(f)
                if content is not None:
                    locations[image_id] = content

        return locations

