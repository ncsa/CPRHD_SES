import os

# create text files for test and train datasets
image_files = []
os.chdir("test")
for filename in os.listdir(xos.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("PATH/TO/TEST/IMAGES/" + filename)
os.chdir("..")
with open("test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()

image_files = []
os.chdir(os.path.join("obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data_1003/obj/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")
