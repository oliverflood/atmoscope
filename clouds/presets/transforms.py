from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TFMS_BASIC = T.Compose([
    T.Resize(256), 
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

TFMS_MEDIUM = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15, hue=0.05),
    T.RandomApply([T.RandomRotation(degrees=10, fill=128)], p=0.5),
    T.ToTensor(),
    T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), inplace=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])