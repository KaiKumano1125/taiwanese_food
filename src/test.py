from torchvision import datasets

train_data = datasets.ImageFolder("../dataset/tw_food_101/train")
labels = [y for _, y in train_data.samples]

print("Min label:", min(labels))
print("Max label:", max(labels))
print("Number of classes detected:", len(train_data.classes))
print("First few class names:", train_data.classes[:10])
