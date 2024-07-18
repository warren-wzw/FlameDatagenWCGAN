import os
import sys
from tkinter import E
import tqdm
import torch
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
from torch.utils.data import (DataLoader)
from datetime import datetime
from model.WCGAN import Generator_Bi,Generator_TConv,Discriminator
from model.utils import FireDataset,load_and_cache_withlabel,get_linear_schedule_with_warmup,PrintModelInfo,clear_directory,visual_result
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
TF_ENABLE_ONEDNN_OPTS=0
BATCH_SIZE=300
EPOCH=5000
LR_G=1
LR_D=1
LR=1e-5
NUM_CLASS=62
PRETRAINED_MODEL_PATH="./output/output_model/cGAN_G_v6_TConv.pth"
TensorBoardStep=500
SAVE_MODEL='./output/output_model/'

"""dataset"""
train_type="train"
image_path_train=f"./dataset/image/{train_type}"
label_path_train=f"./dataset/label/{train_type}/{train_type}.json"
cached_file=f"./dataset/cache/{train_type}_cgan.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradient_penalty(critic, real_data, fake_data, labels, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates.requires_grad_(True)
    d_interpolates,_= critic(interpolates, labels)  
    grads = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def Dynamic_Train(mean_loss_D,mean_loss_G,loss_sumd,loss_sumg):
    DCycleNum=1
    GCycleNum=1
    """D"""
    if loss_sumd-mean_loss_D>0 and DCycleNum<=10:
        DCycleNum=DCycleNum+1
    elif loss_sumd-mean_loss_D<0 and abs(loss_sumd-mean_loss_D)<2:
        DCycleNum=DCycleNum-1
    """G"""
    if loss_sumg-mean_loss_G>0 and GCycleNum<=10:
        GCycleNum=GCycleNum+1
    elif loss_sumg-mean_loss_G<0 and abs(loss_sumg-mean_loss_G)<2:
        GCycleNum=GCycleNum-1
    if GCycleNum==0 and DCycleNum==0:
        DCycleNum=1
    return  DCycleNum,GCycleNum
    
def main():
    c_dim=63
    z_dim=100
    global_step=0 
    GCycleNum=1
    DCycleNum=1
    loss_queue_D=[]
    loss_queue_G=[]
    img_dim = (3, 128, 128) 
    model_G=Generator_TConv(z_dim=z_dim,c_dim=128).to(DEVICE)
    model_D=Discriminator(c_dim,img_dim).to(DEVICE)
    #model_G.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    PrintModelInfo(model_G)
    print()
    PrintModelInfo(model_D)
    dataloader_train=CreateDataloader(image_path_train,label_path_train,cached_file)
    total_steps = len(dataloader_train) * EPOCH
    clear_directory("./output/output_images/")
    """loss"""
    criterion = torch.nn.CrossEntropyLoss()
    """optimizer"""
    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=LR*LR_G)
    optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=LR*LR_D)
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_MODEL}")
    print("  ****************************************************************")
    model_G.train()
    model_D.train()
    scheduler_G = get_linear_schedule_with_warmup(optimizer_G, 0.1 * total_steps , total_steps)
    scheduler_D = get_linear_schedule_with_warmup(optimizer_D, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='./output/tflog/')
    for epoch_index in range(EPOCH):
        loss_sumd=0
        loss_sumdpred=0
        loss_sumg=0
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            #visual_result(image,"output.jpg")
            z = torch.randn(len(image), z_dim, device=DEVICE)
            gen_labels = torch.randint(0, 62, (len(image),), device=DEVICE)
            """train model_D"""
            for i in range(DCycleNum):
                optimizer_D.zero_grad()
                fake_image=model_G(z, gen_labels).detach()
                real_validity,real_labels_pred  = model_D(image, label)
                fake_validity,fake_labels_pred  = model_D(fake_image, gen_labels)
                gp=gradient_penalty(model_D, image, fake_image,label,DEVICE)
                d_loss = - torch.mean(real_validity)+torch.mean(fake_validity)+10*gp
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_D.parameters(), max_norm=2.0)  # 可以调整
                optimizer_D.step()
                for p in model_D.parameters():
                    p.data.clamp_(-0.01, 0.01)
            """train G model"""
            for i in range(GCycleNum):
                optimizer_G.zero_grad()
                gen_imgs = model_G(z, gen_labels)
                fake_validity,fake_labels_pred= model_D(gen_imgs, gen_labels)
                g_loss = -torch.mean(fake_validity)
                loss_sumdpred=0
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=2.0)  # 可以调整
                optimizer_G.step()
            """training detail"""    
            current_lr_G= scheduler_G.get_last_lr()[0]
            current_lr_R= scheduler_D.get_last_lr()[0]
            scheduler_G.step()
            scheduler_D.step()
            loss_sumg=loss_sumg+g_loss.item()
            loss_sumd=loss_sumd+d_loss.item()
            """tqdm"""
            train_iterator.set_description('Epoch=%d, loss_G=%.6f, loss_D=%.6f, lr_G=%9.7f,lr_D=%9.7f,D_NUM=%d,G_NUM=%d'% (
                epoch_index, loss_sumg/(step+1), loss_sumd/(step+1),current_lr_G,current_lr_R,DCycleNum,GCycleNum))
            """ tensorboard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_G/lr', scheduler_G.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_G/loss', g_loss.item(), global_step=global_step)
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_D/lr', scheduler_D.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_D/loss', d_loss.item(), global_step=global_step)
            global_step+=1
        """cal averge loss"""
        loss_queue_D.append(loss_sumd)
        if len(loss_queue_D) > 5:
            loss_queue_D.pop(0) 
                
        loss_queue_G.append(loss_sumg)
        if len(loss_queue_G) > 5:
            loss_queue_G.pop(0)  
        mean_loss_D = sum(loss_queue_D) / len(loss_queue_D) if loss_queue_D else 0.0
        mean_loss_G = sum(loss_queue_G) / len(loss_queue_G) if loss_queue_G else 0.0
        DCycleNum,GCycleNum=Dynamic_Train(mean_loss_D,mean_loss_G,loss_sumd,loss_sumg)
        """Visiual result"""
        SIZE=4*4
        model_G.eval()
        with torch.no_grad():
            z = torch.randn(SIZE, z_dim, device=DEVICE)
            labels = torch.randint(0, NUM_CLASS, (len(image),), device=DEVICE)[:SIZE]
            gen_imgs = model_G(z, labels).detach().cpu()
        fig, axs = plt.subplots(int(SIZE**0.5),int(SIZE**0.5), figsize=(10, 10))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(gen_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)   
            ax.axis('off')
            ax.axis('off')
            label_text = f'Label: {labels[i].item()}'
            ax.text(0.5, -0.1, label_text, fontsize=12, ha='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(f'./output/output_images/realtime.png')   
        plt.close(fig)
                
        if ((epoch_index+1) % 100)==0:  
            SIZE=3*3
            model_G.eval()
            with torch.no_grad():
                z = torch.randn(SIZE, z_dim, device=DEVICE)
                labels = torch.randint(0, NUM_CLASS, (len(image),), device=DEVICE)[:SIZE]
                gen_imgs = model_G(z, labels).detach().cpu()
            fig, axs = plt.subplots(int(SIZE**0.5),int(SIZE**0.5), figsize=(10, 10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(gen_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)   
                ax.axis('off')
                label_text = f'Label: {labels[i].item()}'
                ax.text(0.5, -0.1, label_text, fontsize=12, ha='center', transform=ax.transAxes)
            plt.tight_layout()
            plt.savefig(f'./output/output_images/generated_images_epoch_{epoch_index}.png')   
            plt.close(fig)
              
        if ((epoch_index+1) % 50)==0:
            if not os.path.exists(SAVE_MODEL):
                os.makedirs(SAVE_MODEL)
            torch.save(model_G.state_dict(), os.path.join(SAVE_MODEL,"cGAN_G.pth"))
            print("--->Saving model checkpoint {} at {}".format(SAVE_MODEL+"cGAN_G.pth", 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) 
        torch.cuda.empty_cache() 
        
if __name__ == "__main__":
    main()