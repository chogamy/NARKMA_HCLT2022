import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from nat_base import NATransformer
from mytokenizer import MyTokenizer
from splitter_train import split

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--resume_from_checkpoint ',
                    type=str,
                    help='resume')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train',
                            help='train file')

        parser.add_argument('--valid_file',
                            type=str,
                            default='data/valid',
                            help='valid file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='')
        return parser

class KMADataset(Dataset):
    def __init__(self, filepath, src_tok, morph_tok, tag_tok, max_len, ignore_index=-100) -> None:
        self.filepath = filepath

        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok

        self.max_len = max_len
        self.srcs, self.morphs, self.tags = self.load_data()

        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.srcs)

    def make_enc_input(self, input_ids, tok):
        attention_mask = [1] * len(input_ids) \
                        + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [tok.index("<pad>")] * (self.max_len - len(input_ids))
        
        return input_ids, attention_mask

    def make_dec_input(self, decoder_labels, tok):
        decoder_attention_mask = [1] * len(decoder_labels) \
                                + [0] *(self.max_len - len(decoder_labels))
        decoder_input_ids = [tok.index("<mask>")] * len(decoder_labels) \
                                + [tok.index("<pad>")] * (self.max_len - len(decoder_labels))
        
        return decoder_input_ids, decoder_attention_mask

    def __getitem__(self, index):
        
        # SRC
        src_sent = self.srcs[index]
        src_tokens = list(src_sent) 
        src_tokens.insert(0, "<len>")
        src_id = self.src_tok.encode(src_tokens)
        # [<len>, c1, c2, c3 ...]

        # Morph
        morph_sent = self.morphs[index]
        morph_tokens = list(morph_sent)
        morph_labels = self.morph_tok.encode(morph_tokens)

        # TGT Tag
        tag_tokens = self.tags[index].strip().split(" ")
        tag_labels = self.tag_tok.encode(tag_tokens)

        # LENGTH
        len_labels = len(morph_labels)

        # Module input
        input_ids, attention_mask = self.make_enc_input(src_id, self.src_tok)
        morph_input_ids, morph_attention_mask = self.make_dec_input(morph_labels, self.morph_tok)
        tag_input_ids, tag_attention_mask = self.make_dec_input(morph_labels, self.tag_tok)
        morph_labels = morph_labels + [self.ignore_index] * (self.max_len - len_labels)
        tag_labels = tag_labels + [self.ignore_index] * (self.max_len - len_labels)
        
        
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'attention_mask': np.array(attention_mask, dtype=np.float_),
                'morph_input_ids': np.array(morph_input_ids, dtype=np.int_),
                'morph_attention_mask': np.array(morph_attention_mask, dtype=np.float_),
                'morph_labels': np.array(morph_labels, dtype=np.int_),
                'tag_input_ids': np.array(tag_input_ids, dtype=np.int_),
                'tag_attention_mask': np.array(tag_attention_mask, dtype=np.float_),
                'tag_labels': np.array(tag_labels, dtype=np.int_),
                'len_labels': np.array(len_labels, dtype=np.int_)}

    def load_data(self):
        srcs = []
        morphs = []
        tags = []

        src_f = open(self.filepath + "_src.txt", 'r', encoding="UTF-8-sig")
        morph_f = open(self.filepath + "_morph.txt", 'r', encoding="UTF-8-sig")
        tag_f = open(self.filepath + "_tag.txt", 'r', encoding="UTF-8-sig")

        for src, morph, tag in zip(src_f, morph_f, tag_f):
            src_bufs, morph_bufs, tag_bufs = split(src.strip(), morph.strip(), tag.strip(), self.max_len)

            for src_buf, morph_buf, tag_buf in zip(src_bufs, morph_bufs, tag_bufs):
                srcs.append(src_buf)
                morphs.append(morph_buf)
                tags.append(tag_buf)

        print(len(srcs))    
        assert len(srcs) == len(morphs) == len(tags), "length different"
        return srcs, morphs, tags

class KMAModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, src_tok, morph_tok, tag_tok, max_len, batch_size=8, num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok
        self.max_len = max_len

        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KMADataset(self.train_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)
        self.valid = KMADataset(self.valid_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

class Base(pl.LightningModule):
    def __init__(self, args, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(args)
        self.args = args

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-4,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')

        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=int, default=0.1)
        parser.add_argument("--max_len", type=int, default=256, help="Maximum length of the output utterances")

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class Model(Base):
    def __init__(self, args, **kwargs):
        super(Model, self).__init__(args, **kwargs)
        src_tok = MyTokenizer(extra_special_symbols=["<len>"])
        src_tok.read_vocab(args.train_file + '_src_vocab.txt')
        self.src_tok = src_tok

        morph_tok = MyTokenizer()
        morph_tok.read_vocab(args.train_file + '_morph_vocab.txt')
        self.morph_tok = morph_tok

        tag_tok = MyTokenizer()
        tag_tok.read_vocab(args.train_file +'_tag_vocab.txt')
        self.tag_tok = tag_tok

        self.pad_token_id = src_tok.index("<pad>")

        self.model = NATransformer(args, self.src_tok, self.morph_tok, self.tag_tok)


    def forward(self, inputs):
        return self.model(inputs['input_ids'],
                          inputs['attention_mask'],
                          inputs['morph_input_ids'],
                          inputs['morph_attention_mask'],
                          inputs['morph_labels'],
                          inputs['tag_input_ids'],
                          inputs['tag_attention_mask'],
                          inputs['tag_labels'],
                          inputs['len_labels'])

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs[-1]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs[-1]
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        val_loss_mean = torch.stack(losses).mean()
        self.log('val_loss', val_loss_mean, prog_bar=True)


if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KMAModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = Model(args)

    dm = KMAModule(args.train_file,
                    args.valid_file,
                    model.src_tok, model.morph_tok, model.tag_tok,
                    args.max_len,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    ###
    #early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=200, mode="min")   
    ##

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=None,
                                                       dirpath=args.default_root_dir,
                                                       filename='version_4/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min')
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.gpus, accelerator="dp", logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)    