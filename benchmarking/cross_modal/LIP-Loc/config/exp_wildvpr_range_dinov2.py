import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    debug = False
    batch_size = 64
    num_workers = 32
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = 'Dinov2'
    image_embedding_dim = 384
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.1
    # image size
    size = 518
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    crop = True
    dataloader = "WildVPRDatasetRangeView"
    model = "CLIPModelV1NormNewLoss"
    use_class_token = True
    
    # Cropping
    crop = True
    crop_distance=False
    distance_threshold=30.0
