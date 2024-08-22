import torch
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be evaluated")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load the model
    model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP")
    model.eval().to(device)

    # Define CLIP's normalization transform
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # Load the image
    img = Image.open(args.img_path).convert("RGB")

    # Preprocess the images
    img = transforms.ToTensor()(img)
    img = normalize(img).to(device)

    # Compute the quality score
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img)

    print(f"Image {args.img_path} quality score: {score.item()}")
