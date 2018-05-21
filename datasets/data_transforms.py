from datasets import transforms_ext
from torchvision import transforms


def get_val_transform(image_size=224):
    val_transform = transforms.Compose([
                transforms_ext.Resize(w=image_size, h=image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    return val_transform

def get_train_aug_transform(image_size=224):
    train_aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms_ext.Resize(w=image_size, h=image_size),
        transforms.ToTensor(),
        transforms_ext.ColorJitter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_aug_transform


