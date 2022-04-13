import os
import json
import logging
import argparse
import torch

from source.inputter.corpus import KnowledgeCorpus
from source.model.seq2seq import Seq2Seq
from source.utils.engine import Trainer
from source.utils.generator_csv import BeamGenerator
from source.utils.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./prepared_data/cmu_DoG")
    data_arg.add_argument("--save_dir", type=str, default="./models/cmu_new")
    data_arg.add_argument("--output_dir", type=str, default="./outputs/cmu_new")
    data_arg.add_argument("--vocab_type", type=str, default="all")
    data_arg.add_argument("--embed_file", type=str, default=None)

    # Network
    # change args according to params.json
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=256)
    net_arg.add_argument("--bidirectional", type=str2bool, default=False)
    net_arg.add_argument("--max_vocab_size", type=int, default=30000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=48)
    net_arg.add_argument("--min_kb_len", type=int, default=0)
    net_arg.add_argument("--max_kb_len", type=int, default=98)
    net_arg.add_argument("--num_attn_layers", type=int, default=2)
    net_arg.add_argument("--num_rnn_layers", type=int, default=1)
    net_arg.add_argument("--max_dlg_hop", type=int, default=3)
    net_arg.add_argument("--max_kb_hop", type=int, default=3)
    net_arg.add_argument("--attn", type=str, default='mlp', choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=False)
    net_arg.add_argument("--with_bridge", type=str2bool, default=False)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=False)

    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--gpu", type=int, default=3)
    train_arg.add_argument("--batch_size", type=int, default=16)
    train_arg.add_argument("--window_size", type=int, default=3)
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--patience", type=int, default=5)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.2)
    train_arg.add_argument("--num_epochs", type=int, default=30)
    train_arg.add_argument("--pre_epochs", type=int, default=10)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)
    train_arg.add_argument("--num_heads", type=int, default=8)
    train_arg.add_argument("--pf_dim", type=float, default=2048)
    train_arg.add_argument("--log_steps", type=int, default=100)
    train_arg.add_argument("--valid_steps", type=int, default=500)
    train_arg.add_argument("--fp16", action='store_true')
    train_arg.add_argument("--fp16_opt_level", type=str, default='O1',
                            help="See details at https://nvidia.github.io/apex/amp.html")
    train_arg.add_argument("--grad_accum_steps", type=int, default=1)
    train_arg.add_argument("--n_gpu", type=int, default=1)
    train_arg.add_argument("--bert_config", type=str, default="config.json")

    # Generation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--test", action="store_true")
    gen_arg.add_argument("--ckpt", type=str, default="best.model")
    gen_arg.add_argument("--beam_size", type=int, default=4)
    gen_arg.add_argument("--max_dec_len", type=int, default=20)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)

    config = parser.parse_args()
    return config

def main():
    """
    main
    """
    config = model_config()
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0

    # Data definition
    corpus = KnowledgeCorpus(data_dir=config.data_dir,
                             min_freq=0, max_vocab_size=config.max_vocab_size,
                             min_len=config.min_len, max_len=config.max_len, min_kb_len=config.min_kb_len, max_kb_len=config.max_kb_len,
                             embed_file=config.embed_file, share_vocab=config.share_vocab, vocab_type=config.vocab_type)
    corpus.load()
    print('corpus loaded')

    if config.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Model definition
    model = Seq2Seq(src_field=corpus.SRC, tgt_field=corpus.TGT,
                    kb_field=corpus.KB, kbt_field = corpus.KBT, embed_size=config.embed_size,
                    hidden_size=config.hidden_size, padding_idx=corpus.padding_idx,
                    num_attn_layers=config.num_attn_layers, num_rnn_layers=config.num_rnn_layers, 
                    bidirectional=config.bidirectional,
                    attn_mode=config.attn, with_bridge=config.with_bridge,
                    tie_embedding=config.tie_embedding, dropout=config.dropout,
                    max_dlg_hop=config.max_dlg_hop, max_kb_hop=config.max_kb_hop, use_gpu=config.use_gpu,
                    pf_dim=config.pf_dim, num_heads=config.num_heads, bert_config=config.bert_config, window_size=config.window_size)

    # Generator definition
    generator = BeamGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT,
                              kb_field=corpus.KB, kbt_field = corpus.KBT, beam_size=config.beam_size, max_length=config.max_dec_len,
                              ignore_unk=config.ignore_unk,
                              length_average=config.length_average, use_gpu=config.use_gpu)

    # Testing
    if config.test and config.ckpt:
        test_iter = corpus.create_batches(config.batch_size, data_type="test", shuffle=False)

        model_path = os.path.join(config.save_dir, config.ckpt)
        model.load(model_path)
        print("Testing ...")
        metrics = Trainer.evaluate(model, test_iter, generator= generator)
        print(metrics.report_cum())
        print("Generating ...")
        generator.generate(data_iter=test_iter, output_dir=config.output_dir, verbos=True)

    else:
        train_iter = corpus.create_batches(config.batch_size, data_type="train", shuffle=True)
        valid_iter = corpus.create_batches(config.batch_size, data_type="valid", shuffle=False)

        print("Iterators done")
        
        # Load word embeddings if possible
        if config.use_embed and config.embed_file is not None:
            model.rnn_encoder.embedder.load_embeddings(corpus.SRC.embeddings, scale=0.03)
            model.decoder.embedder.load_embeddings(corpus.TGT.embeddings, scale=0.03)

        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)

        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=config.lr_decay,
                patience=config.patience, verbose=True, min_lr=1e-6)
        else:
            lr_scheduler = None

        # Save directory
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        logger.info("Saved params to '{}'".format(params_file))
        logger.info(model)


        # Training
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, valid_metric_name="-loss",
                          num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler, entity_dir=config.data_dir, generator=generator)
        
        if config.ckpt is not None:
            trainer.load(file_ckpt=config.ckpt)
        trainer.train()

        logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
