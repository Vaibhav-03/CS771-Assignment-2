import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import (PatchEmbed, window_partition, window_unpartition,
                           DropPath, MLP, trunc_normal_)


################################################################################
# You will need to fill in the missing code in this file
################################################################################


################################################################################
# Part I.1: Understanding Convolutions
################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation. We only consider square
        filters with equal stride and padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input.
            Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        ########################################################################
        # Fill in the code here
        
        # Add padding to input if needed
        if padding > 0:
            input_padded = torch.nn.functional.pad(input_feats, (padding, padding, padding, padding))
        else:
            input_padded = input_feats
        
        # Unfold the input to get patches
        patches = unfold(input_padded, kernel_size, stride=stride)  # (N, C_i * K * K, H_out * W_out)
        
        # Reshape patches for matrix multiplication: (N, H_out * W_out, C_i * K * K)
        patches_reshaped = patches.permute(0, 2, 1)  # (N, H_out * W_out, C_i * K * K)
        
        # Reshape weight for matrix multiplication
        weight_reshaped = weight.view(weight.size(0), -1)  # (C_o, C_i * K * K)
        
        # Perform convolution by matrix multiplication
        output_patches = torch.matmul(patches_reshaped, weight_reshaped.t())  # (N, H_out * W_out, C_o)
        
        # Add bias if provided
        if bias is not None:
            output_patches = output_patches + bias
        
        # Calculate output dimensions
        H_out = (ctx.input_height + 2 * padding - kernel_size) // stride + 1
        W_out = (ctx.input_width + 2 * padding - kernel_size) // stride + 1
        
        # Reshape and fold back to get output
        output = output_patches.permute(0, 2, 1).contiguous().view(
            input_feats.size(0), weight.size(0), H_out, W_out
        )

        # Important: return a cloned tensor to avoid returning a view from a
        # custom autograd Function. Returning a view can cause issues when
        # subsequent in-place ops (e.g., ReLU(inplace=True)) are applied and
        # PyTorch's autograd can't reconcile view+inplace with custom backward.
        output = output.clone()
        
        ########################################################################

        # save for backward (you need to save the unfolded tensor into ctx)
        ctx.save_for_backward(patches_reshaped, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term

        """
        # unpack tensors and initialize the grads
        patches_reshaped, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width

        ########################################################################
        # Fill in the code here
        
        # Reshape grad_output for matrix operations
        grad_output_reshaped = grad_output.view(grad_output.size(0), grad_output.size(1), -1)  # (N, C_o, H_out * W_out)
        grad_output_reshaped = grad_output_reshaped.permute(0, 2, 1)  # (N, H_out * W_out, C_o)
        
        # Compute gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output * input_patches
            grad_weight = torch.matmul(patches_reshaped.transpose(1, 2), grad_output_reshaped)  # (N, C_i * K * K, C_o)
            grad_weight = grad_weight.sum(0).permute(1, 0).view(weight.size())  # (C_o, C_i, K, K)
        
        # Compute gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output * weight
            weight_reshaped = weight.view(weight.size(0), -1)  # (C_o, C_i * K * K)
            grad_patches = torch.matmul(grad_output_reshaped, weight_reshaped)  # (N, H_out * W_out, C_i * K * K)
            grad_patches = grad_patches.permute(0, 2, 1)  # (N, C_i * K * K, H_out * W_out)

            # Fold back to get grad_input
            H_out = (input_height + 2 * padding - kernel_size) // stride + 1
            W_out = (input_width + 2 * padding - kernel_size) // stride + 1
            grad_input_padded = fold(
                grad_patches,
                (input_height + 2 * padding, input_width + 2 * padding),
                kernel_size,
                stride=stride,
            )

            # Remove padding to get final grad_input
            if padding > 0:
                grad_input = grad_input_padded[:, :, padding:-padding, padding:-padding].contiguous()
            else:
                grad_input = grad_input_padded.contiguous()
        else:
            grad_input = None
        
        ########################################################################
        # compute the gradients w.r.t. input and params

        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(
            input, self.weight, self.bias, self.stride, self.padding
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


################################################################################
# Part I.2: Design and train a convolutional network
################################################################################
class Bottleneck(nn.Module):
    def __init__(self, conv_op, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv_op(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv_op(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = conv_op(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=False)

        
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                conv_op(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out

class SimpleNet(nn.Module):
    # improved simple CNN for image classification
    def __init__(self, conv_op=nn.Conv2d, num_classes=100, dropout=0.0):
        super(SimpleNet, self).__init__()

        # Stem: use a lighter 3x3 stack instead of a heavy 7x7
        self.stem = nn.Sequential(
            conv_op(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            conv_op(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            conv_op(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        
        self.layer2 = Bottleneck(conv_op, in_ch=64,  mid_ch=64,  out_ch=256, stride=1)
        self.pool2  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer3 = Bottleneck(conv_op, in_ch=256, mid_ch=128, out_ch=512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, adv_eps=1e-4):
        if self.training:
            x_adv = x.detach().clone().requires_grad_(True)
            self.eval()
            y = self.stem(x_adv)
            y = self.layer2(y)
            y = self.pool2(y)
            y = self.layer3(y)
            y = self.avgpool(y)
            y = torch.flatten(y, 1)
            y = self.dropout(y)
            logits = self.fc(y)

            (B, C) = logits.shape 
            targets = torch.randint(1, C, (B,)).to(logits.device)
            loss = nn.CrossEntropyLoss()(logits, targets)
            grad_x = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                x_adv = x_adv + adv_eps * grad_x.sign()
                delta = torch.clamp (x_adv - x, min = -adv_eps, max = adv_eps)
                x = torch.clamp(x + delta, min = 0, max = 1).detach()

        x = self.stem(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# change this to your model!
default_cnn_model = SimpleNet

################################################################################
# Part II.1: Understanding self-attention and Transformer block
################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels. We assume Q, K, V will be of
                same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Accept either (B, H, W, C) or (B, N, C) where N = H*W or num tokens
        orig_ndim = x.dim()
        if x.dim() == 4:
            B, H, W, C = x.shape
            x_flat = x.view(B, H * W, C)
            N = H * W
            restore_shape = (B, H, W, C)
        elif x.dim() == 3:
            B, N, C = x.shape
            x_flat = x
            restore_shape = None
        else:
            raise ValueError(f"Unsupported input dims to Attention: {x.dim()}")

        # qkv with shape (3, B, nHead, N, C_head)
        qkv = (
            self.qkv(x_flat)
            .reshape(B, N, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        ########################################################################
        # Fill in the code here
        ########################################################################
        att = (q @ k.transpose(-2, -1)) * self.scale # (B * nHead, H * W, H * W)
        att = torch.nn.functional.softmax(att, dim=-1) # (B * nHead, H * W, H * W)
        x = (att @ v) # (B * nHead, H * W, C)
        x = x.view(B, self.num_heads, H * W, -1) # (B, nHead, H * W, C)
        x = x.permute(0, 2, 1, 3).reshape(B, H * W, -1) # (B, H * W, nHead * C)
        x = self.proj(x).view(B, H, W, -1) # (B, H, W, C)
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        ########################################################################
        # Fill in the code here
        ########################################################################
        if self.window_size > 0:
            _, H, W, _ = x.shape
            # partition windows
            x_windows, pad_hw = window_partition(x, self.window_size) #[B * num_windows, window_size, window_size, C], (Hp, Wp)
            attn_windows = self.attn(x_windows)  # (B * num_windows, window_size, window_size, C)
            x = window_unpartition(attn_windows, self.window_size, pad_hw, (H, W))  # (B, H, W, C)
        else:
            #global attention
            x = self.attn(x)

        # The implementation shall support local self-attention
        # (also known as window attention)

        # MLP after MSA, both can be dropped at random
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#################################################################################
# Part II.2: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and
                can be coupled with standard global attention. E.g., [0, 2]
                indicates the first and the third blocks will use local window
                attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall define some Transformer blocks
        blocks = []
        for i in range(depth):
            layer_window_size=0
            if i in window_block_indexes:
                layer_window_size = window_size
            blocks.append(TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dpr[i], norm_layer, act_layer, layer_window_size))
        self.transformer_blocks = nn.Sequential(*blocks)
        self.out=nn.Linear(embed_dim, num_classes)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Fill in the code here
        ########################################################################
        x = self.patch_embed(x)
        x = x + self.pos_embed if self.pos_embed is not None else x
        x = self.transformer_blocks(x)
        x = x.mean(dim=(1,2)) # global average pooling
        x = self.out(x)
        return x

# change this to your model!
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    train_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms():
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    val_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


################################################################################
# Part III: Adversarial samples
################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.01):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    
    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False
        # loop over the number of steps
        # for _ in range(self.num_steps):
        with torch.no_grad():
            logits = model(input)
            least_conf_label = logits.argmin(dim=1)

        ########################################################################
        adv_input = input.clone().detach()
        for _ in range(self.num_steps): 
            adv_input.requires_grad = True
            output = model(adv_input)
            loss = torch.nn.CrossEntropyLoss()(output, least_conf_label)
            grad = torch.autograd.grad(loss, adv_input, retain_graph=False, create_graph=False)[0]
            adv_input = adv_input + self.step_size * grad.sign()
            delta = torch.clamp (adv_input - input, min = -self.epsilon, max = self.epsilon)
            adv_input = torch.clamp(input + delta, min = 0, max = 1).detach()

        output = adv_input         
        ########################################################################
        return output
    
default_attack = PGDAttack


def vis_grid(input, n_rows=10):
    """
    Given a batch of image X (torch tensor), compose a mosaic for visualziation.

    Args:
      input: (torch tensor) input image of size N * C * H * W
      n_rows: (int) number of images per row

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    output_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    return output_imgs

default_visfunction = vis_grid
