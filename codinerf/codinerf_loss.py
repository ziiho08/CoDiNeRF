import open_clip
from PIL import Image
import torch
from torch import Tensor, nn
from typing import Literal
import cv2
L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

class clip_loss(nn.Module):
    
    def reshape_(self, output):
        #output = output.reshape(540, 960, 3)
        output = output[:1024, :]
        output = output.view(32,32,3) 
        output = torch.moveaxis(output, -1, 0)[None, ...]
        output = output.squeeze(0)  # Remove the batch dimension
        output = output.permute(1, 2, 0)  
        output = output.detach().cpu().numpy()
        output = (output*255).astype('uint8')
        pil_output = Image.fromarray(output)

        return pil_output       
    
    def __init__(self, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b')
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')

    def forward(self, gt_rgb, pred_rgb):
        # pred_rgb = pred_rgb.view(64,64,3)
        # pred_rgb = torch.moveaxis(pred_rgb, -1, 0)[None, ...]
        # print(pred_rgb.shape)
        #image = self.preprocess((np.uint16(pred_rgb.cpu().detach().numpy()))).unsqueeze(0).to(self.device)
        #print("cliploss forward", image.shape) #  torch.Size([1, 3, 224, 224])
        # print("forward pred unique", np.unique((image.cpu().detach().numpy())))
        #print("pred_rgb", np.unique(pred_rgb))
        #plt.imsave("/hdd/pred_rgb.png", Image.fromarray(self.reshape_(pred_rgb)))
        
        pred_image = self.preprocess(self.reshape_(pred_rgb)).unsqueeze(0).to(self.device) # torch.Size([1, 3, 224, 224])
        gt_image = self.preprocess(self.reshape_(gt_rgb)).unsqueeze(0).to(self.device)
        
        save_pred = gt_image.squeeze(0)
        save_pred = save_pred.permute(1, 2, 0)  # Change to (H, W, C)
        save_pred = save_pred.cpu().detach().numpy()
        #cv2.imwrite('/hdd/output_image.png', save_pred*255)
    
        text = self.tokenizer(["A cropped photo of the " + desc for desc in ["car","sky","road","person","traffic sign","sidewalk","buildings","vegetation"]]).to(self.device) # [1,77]
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred_embeddings = self.model.encode_image(pred_image) #[1,512]
            gt_embeddings = self.model.encode_image(gt_image) #[1,512]
            text_embeddings = self.model.encode_text(text) #[1,512]

            pred_embeddings_ = pred_embeddings / pred_embeddings.norm(dim=-1, keepdim=True)
            gt_embeddings_ = gt_embeddings / gt_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings_ = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            pred_similarity = pred_embeddings_ @ text_embeddings_.T
            gt_similarity = gt_embeddings_ @ text_embeddings_.T

        self.mse_loss = MSELoss(reduction="mean")
        self.L1_loss = L1Loss()

        sim_loss = self.L1_loss((100* pred_similarity).softmax(dim=-1), (100*gt_similarity).softmax(dim=-1))

        return sim_loss
    
