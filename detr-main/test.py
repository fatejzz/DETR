import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.append("/home/hkyunqi/jzz/DETR/detr-main")
from main import get_args_parser

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model




device = torch.device(0)
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args(args=[])
model, criterion, postprocessors = build_model(args)

args.distributed = False
model_without_ddp = model
param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                              weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

dataset_train = build_dataset(image_set='train', args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_train)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=args.num_workers)

'''backbone'''
from models.backbone import  *
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
backbone = Backbone(args.backbone, True, False, args.dilation)
position_embedding = build_position_encoding(args)
model_joiner=Joiner(backbone, position_embedding)

'''transformer'''
from models.transformer import *
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,   ##是否保存解码层中间层的结果
    )
transformer = build_transformer(args)


for data in data_loader_train:
    # x,y=model_joiner(data[0])
    outputs=model(data[0])

##matcher
targets=data[1]
bs, num_queries = outputs["pred_logits"].shape[:2]
# We flatten to compute the cost matrices in a batch
out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
# Also concat the target labels and boxes
tgt_ids = torch.cat([v["labels"] for v in targets]) ##各tgt框的类别信息
tgt_bbox = torch.cat([v["boxes"] for v in targets])
cost_class = -out_prob[:, tgt_ids] ##类别代价
cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)##box l1代价
# Compute the giou cost betwen boxes
from util.box_ops import *
cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
C = 1 * cost_bbox + 1 * cost_class + 1 * cost_giou
C = C.view(bs, num_queries, -1).cpu()
from scipy.optimize import linear_sum_assignment
sizes = [len(v["boxes"]) for v in targets]
indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

