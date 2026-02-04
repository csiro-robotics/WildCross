import timm
import torch.nn.functional as F
from torch import nn
import torch

class DinoWrapperV2(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino 
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches


    def forward(self, x):
        B, _, H, W = x.shape
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
        # train all blocks
        for blk in self.dino.blocks:
            x = blk(x)

        # Get class token
        x = x[:,0]
        return x 


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable
    ):
        super().__init__()
        if model_name == 'Dinov2':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model = DinoWrapperV2(dino)
            
        elif model_name == 'Dinov3':
            # Here define REPO_DIR and CHECKPOINT_PATH to your local dinov3 repo and checkpoint paths
            # See https://github.com/facebookresearch/dinov3 for more details
            REPO_DIR="/path/to/dinov3"
            CHECKPOINT_PATH="/path/to/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            
            dino = torch.hub.load(REPO_DIR,
                                'dinov3_vits16',
                                source='local',
                                weights=CHECKPOINT_PATH)
        
            self.model = dino
        
        
        else:
            self.model = timm.create_model(
                model_name, pretrained, num_classes=0, global_pool="avg"
            )
            for p in self.model.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.device=CFG.device
        self.temperature=CFG.temperature
        self.encoder_camera = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.encoder_lidar = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.projection_lidar = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        self.projection_camera = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.neg_thresh = 50.0

    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        camera_image_features = self.encoder_camera(batch["camera_image"])
        lidar_image_features = self.encoder_lidar(batch["lidar_image"])
                
        # Getting camera Image and lidar range Embeddings (with same dimension)
        camera_image_embeddings = self.projection_camera(camera_image_features)
        lidar_image_embeddings = self.projection_lidar(lidar_image_features)

        # Normalize the embeddings to have unit norm
        camera_image_embeddings = F.normalize(camera_image_embeddings, dim=-1)
        lidar_image_embeddings = F.normalize(lidar_image_embeddings, dim=-1)

        # Calculating the Loss
        # Getting ignore matrix 
        cross_dist = torch.cdist(batch['lidar_coord'], batch['lidar_coord'])
        negative_mask = cross_dist < self.neg_thresh
        negative_mask.fill_diagonal_(False)

        
        logits = (lidar_image_embeddings @ camera_image_embeddings.T) / self.temperature
        logits[negative_mask] = -10000.0 # Set non-negatives to a low number so they're ignored when calculating softmax 
        
        targets = torch.arange(len(logits), device='cuda')
        lidar_loss = self.cross_entropy(logits, targets)
        camera_loss = self.cross_entropy(logits.T, targets)
        
        loss_total = (lidar_loss + camera_loss) / 2.0 
        
        # Calculate number past filter 
        with torch.no_grad():
            lidar_softmax_logits = F.softmax(logits, dim = -1)
            camera_softmax_logits = F.softmax(logits.T, dim = -1)
            
            average_diagonal = (torch.diagonal(lidar_softmax_logits) + torch.diagonal(camera_softmax_logits)) / 2.0
            num_past_filter = torch.sum(average_diagonal >= 0.95) / len(average_diagonal)
        
        return loss_total, num_past_filter

    def get_camera_embeddings(self, batch):
        image_features = self.encoder_camera(batch["camera_image"].to(self.device))
        image_embeddings = self.projection_camera(image_features)

        # Normalize the embeddings to have unit norm
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        return image_embeddings

    def get_lidar_embeddings(self, batch):
        image_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        image_embeddings = self.projection_lidar(image_features)

        # Normalize the embeddings to have unit norm
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        return image_embeddings
