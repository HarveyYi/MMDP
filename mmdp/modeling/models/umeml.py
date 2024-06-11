import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint



from mmdp.modeling.ops import (SNN_Block, MultiheadAttention, TransLayer, BilinearFusion, 
                                compute_modularity)

from .build import MODEL_REGISTRY
from .base import Base



def reset(x, n_c): x.data.uniform_(-1.0 / n_c, 1.0 / n_c)




class PathProtoGenerator(nn.Module):
    def __init__(
            self,
            dim: int,
            drop_path: float = 0.,
    ) -> None:
        super().__init__()
        self.cross_attn = MultiheadAttention(embed_dim=dim, num_heads=1)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _c, attn = self.cross_attn(c.transpose(1, 0), x.transpose(1, 0), x.transpose(1, 0),)  # ([5, 1, 256])
        _c = _c.transpose(1, 0)
        c = c + self.drop_path1(self.norm1(_c))
        return c





class Block(nn.Module):
    def __init__(
            self,
            dim: int,
    ) -> None:
        super().__init__()
        self.attn = TransLayer(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return x


    
class BottleneckAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int=256,
            n_reg: int=2,
    ) -> None:
        super().__init__() 
        self.bottle_tokens = nn.Parameter(torch.FloatTensor(1, n_reg, dim), requires_grad=True)
        nn.init.uniform_(self.bottle_tokens)
        self.encoders =  nn.ModuleList([
           Block(dim=dim)
            for i in range(2)])
        
        
    def forward(self, x_path: torch.Tensor, x_omic: torch.Tensor):
        # import pdb;pdb.set_trace()
        path_len,  omic_len = x_path.size()[1],  x_omic.size()[1]
        
        token_len = self.bottle_tokens.size()[1]
            
        x = torch.concat([x_path, self.bottle_tokens, x_omic], dim=1)
        for blk in self.encoders:
            x = blk(x)
        t_path, x_path = x[:, :1, :], x[:,  1 :(path_len), :]
        t_omic, x_omic = x[:, (path_len + token_len): (path_len + token_len + 1), : ], x[:, (path_len + token_len + 1): , : ]
        return t_path, x_path, t_omic, x_omic
                 

class UMEML(Base):
    def __init__(self,
                 cfg, 
                 num_classes,
                 omic_sizes,
                 ):
        super(UMEML, self).__init__()
        self.cfg = cfg
        self.root = osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT))
        
        dropout = self.cfg.MODEL.DROPOUT
        path_input_dim = self.cfg.DATASET.PATH.DIM
        omic_input_dim = self.cfg.DATASET.OMIC.DIM
        hidden_dim = self.cfg.MODEL.HIDDEN_DIM  # 256
        projection_dim = self.cfg.MODEL.PROJECT_DIM  # 256
        self.fusion = self.cfg.MODEL.FUSION
        self.size = self.cfg.MODEL.SIZE
        
        self.n_proto = self.cfg.MODEL.UMEML.PROTOTYPES 
        self.n_reg = self.cfg.MODEL.UMEML.REGISTERS 
        
        

        p_fc = [nn.Linear(path_input_dim, hidden_dim), nn.ReLU()]
        p_fc.append(nn.Dropout(dropout))
        self.path_net = nn.Sequential(*p_fc)
        
        o_fc = [nn.Linear(omic_input_dim, hidden_dim), nn.ReLU()]
        o_fc.append(nn.Dropout(dropout))
        self.omic_net = nn.Sequential(*o_fc)
        
        g_o_fc = [nn.Linear(1000, hidden_dim), nn.ReLU()]
        g_o_fc.append(nn.Dropout(dropout))
        self.g_omic_net = nn.Sequential(*g_o_fc)
        
        self.proto_g_blocks = nn.ModuleList([
            PathProtoGenerator(dim=hidden_dim)
            for i in range(2)])

        omic_encoder = nn.ModuleList([
            Block(dim=hidden_dim)
            for i in range(2)])
        self.omic_encoder = nn.Sequential(*omic_encoder)

        self.layer_norm_p = nn.LayerNorm(hidden_dim)
        self.layer_norm_o = nn.LayerNorm(hidden_dim)

        self.path_decoder = TransLayer(dim=hidden_dim)
        self.omic_decoder = TransLayer(dim=hidden_dim) 
           
        self.bottleattn = BottleneckAttentionBlock(dim=hidden_dim, n_reg=self.n_reg)
    
        
        self.p_proto = nn.Parameter(torch.empty(1, self.n_proto, hidden_dim))
        reset(self.p_proto, self.n_proto)
        
        self.p_encoder_token = nn.Parameter(torch.FloatTensor(1, 1, hidden_dim), requires_grad=True)
        nn.init.uniform_(self.p_encoder_token)
        
        self.o_encoder_token = nn.Parameter(torch.FloatTensor(1, 1, hidden_dim), requires_grad=True)
        nn.init.uniform_(self.o_encoder_token)    

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=hidden_dim, dim2=hidden_dim, scale_dim1=8, scale_dim2=8, mmhid=hidden_dim)
        else:
            self.mm = None
            
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, x_path, x_omic):
        bsz = x_path.shape[0]
        _, N = x_omic.size()
        x_omic = x_omic.reshape(bsz, -1, N)
        g_omic = x_omic.detach().clone()
        
        x_omic =  x_omic.reshape(bsz, -1, self.omic_input_dim)
        h_path_bag = self.path_net(x_path) 
        # h_path_bag = x_path
        h_omic_bag = self.omic_net(x_omic)
        g_omic = self.g_omic_net(g_omic)
        
        h_omic_bag = torch.concat([h_omic_bag, g_omic], dim=1)

        # import pdb;pdb.set_trace()
        for i, blk in enumerate(self.proto_g_blocks):
            if i == 0:
                p_proto = blk(h_path_bag, self.p_proto)
            else:
                p_proto = blk(h_path_bag, p_proto)
         
        h_omic = torch.concat([self.o_encoder_token, h_omic_bag], dim=1)       
                
        h_omic = self.omic_encoder(h_omic)
      
        h_path = torch.concat([self.p_encoder_token, p_proto], dim=1)
        
        h_path = self.path_decoder(h_path)
        h_omic = self.omic_decoder(h_omic)
        
        h_path = self.layer_norm_p(h_path)
        h_omic = self.layer_norm_o(h_omic)

        t_path, f_path, t_omic, f_omic = self.bottleattn(h_path, h_omic)

        if self.training:        
            modular_1 = compute_modularity(p_proto, h_path_bag, grid=False)
            modular_2 = compute_modularity(h_omic, h_path_bag, grid=False)
            modular_loss =  (modular_1 +  modular_2)

        else:
            modular_loss = 0
            
            
        if self.fusion == 'bilinear':
            h = self.mm(t_path[0], t_omic[0])
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([t_path[0], t_omic[0]], axis=1))

        logits = self.classifier(h)
        
        if self.training: 
            return logits, modular_loss
        else:
            return logits




@MODEL_REGISTRY.register()
def umeml(**kwargs):
    return UMEML(**kwargs)