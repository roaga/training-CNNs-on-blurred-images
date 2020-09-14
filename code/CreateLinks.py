import os

classSet = ['n02877765_', 'n02283201_', 'n01694178_', 'n01985128_', 'n02917067_', 'n01742172_', 'n02910353_', 'n02708093_', 'n01491361_', 'n02114367_', 'n02489166_', 'n02138441_', 'n01770081_', 'n02481823_', 'n02997607_', 'n02807133_', 'n01774384_', 'n01728572_', 'n01664065_', 'n02870880_', 'n01740131_', 'n01729322_', 'n02510455_', 'n02268443_']

blurLevels = [23, 11, 5, 3, 1]
root_dir = "/home/cnslab/imagenet_blur/"

train = 1 # used to create a rough 50/50 split since it alternates every time
for i in range(len(blurLevels)): # different levels of blurring
    mult = blurLevels[i]
    print("Working with square kernel dimension = " + str(mult))
    for directory, subdirectories, files in os.walk(root_dir):
        endPath = directory.split("/")[-1]
        if endPath in classSet:
            for file in files:
                if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                    name = os.path.join(directory, file)
                    if train == 1:
                        if not os.path.exists('/home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'train/' + endPath):
                            os.system('mkdir /home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'train/' + endPath)
                        os.system("ln -s " + name + '/home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'train/' + endPath + '/' + file)
                        train *= -1
                    else:
                        if not os.path.exists('/home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'val/' + endPath):
                            os.system('mkdir /home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'val/' + endPath)
                        os.system("ln -s " + name + '/home/cnslab/imagenet_blur24/gaussian/imagenet_' + str(mult) + '/' + 'val/' + endPath + '/' + file)
                        train *= -1
