import os
import sys
import torch
import torchvision
from torchvision.utils import save_image
os.chdir(sys.path[0])
from model.WCGAN import Generator_TConv
from model.utils import clear_directory,visual_result
TF_ENABLE_ONEDNN_OPTS=0
MODEL_PATH="./output/output_model/cGAN_G_v5_TConv.pth"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    c_dim=63
    z_dim=100
    model_G=Generator_TConv(z_dim=z_dim,c_dim=128).to(DEVICE)
    model_G.load_state_dict(torch.load(MODEL_PATH),strict=False)
    clear_directory("./output/Gen_image/")
    
    model_G.eval()
    with torch.no_grad():
        z = torch.randn(c_dim, z_dim, device=DEVICE)  # Generate z for all 63 classes
        labels = torch.arange(c_dim, device=DEVICE)  # Generate labels from 0 to 62
        gen_imgs = model_G(z, labels).detach().cpu()  # Generate images
        #visual_result(gen_imgs.detach(),"output.jpg")
        # Save generated images
        for idx, img in enumerate(gen_imgs):
            img = img.permute(1, 2, 0) * 0.5 + 0.5  # Permute and normalize
            img = img.numpy()  # Convert to numpy array
            save_path = f"./output/Gen_image/img_{idx}.png"
            save_image(torch.tensor(img).permute(2, 0, 1), save_path)  # Save the image

    
if __name__=="__main__":
    main()