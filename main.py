import imagehandler
import neuralnetwork
import helper
import customdataloader
import torch.utils.data as data_loader
import torchvision.models as pretrained_models
import torch.nn as nn
import torch.optim as optim
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Program started on device", str(device))

trainingsset = "/Trainingsset"
testdir = "/Testset"

#imagehandler.change_image_format("/Images", ".png")
#imagehandler.create_directories_for_class("/classification_zilien_eval.csv", "/CiliaSharp/", trainingsset, types_cilia)
#imagehandler.rotate_images_in_folder(trainingsset+'bad_cilia', 270)

transform_helper = helper.TransformationHelper()

# load testset in the pytorch dataloader
testdir = os.getcwd() + testdir
testset = customdataloader.MyImageFolder(testdir, transform_helper.get_test_transformations())
testset_loader = data_loader.DataLoader(testset, shuffle=True, num_workers=4)

# load trainingsdata in the pytorch dataloader
data_dir = os.getcwd() + trainingsset
dset = customdataloader.MyImageFolder(data_dir, transform_helper.get_trainings_transformations())
dset_loader = data_loader.DataLoader(dset, shuffle=True, num_workers=4, batch_size=4)
dset_classes = dset.classes

net = pretrained_models.resnet50(pretrained=True)

ct = 0
for child in net.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
net.fc = torch.nn.Linear(2048, len(dset_classes))
#net.load_state_dict(torch.load(os.getcwd() + '/serialized_nets/ResNet18SimilarPlaces_v1000'))
neuralnetwork.train_network(net, dset_loader, testset, 1000, optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001), nn.CrossEntropyLoss(), device)

print("Training finished")
tester = helper.TestHelper(testset_loader, testset.classes, net, device)
tester.test_total_precision()
tester.print_total_precision("ResNet34", "Finished")

#json_locations_helper = helper.JsonLocationLoader("json", dset_classes)
#locations_dict = json_locations_helper.load_all_locations()
#for loc in locations_dict:
#    print(loc, ":", len(locations_dict[loc]))
#    print(locations_dict[loc].keys())
