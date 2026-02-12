from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from models.ECRTM.ECRTM import ECRTM
from models.FASTOPIC.FASTOPIC import FASTOPIC
from models.NSTM.NSTM import NSTM
from models.CTM import CTM
from models.ETM import ETM
from models.ProdLDA import ProdLDA
from models.WETE import WeTe
from models.SAE_NTM import SAE
import datasethandler
import scipy
from evaluate import evaluate
import torch
import copy
# import wandb

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    config.add_wete_argument(parser)
    args = parser.parse_args()
    
    prj = args.wandb_prj if args.wandb_prj else 'baselines'

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" + str(args.dataset) + "/" 
                                   + str(args.num_topics),str(args.weight_ECR)+"-"+str(args.epochs)+"-"+current_time)
    current_checkpoint_dir = os.path.join(current_run_dir, "checkpoints")
    use_dpo = args.enable_dpo and args.model == "ECRTM"
    base_content_dir = None
    base_checkpoint_dir = current_checkpoint_dir
    if use_dpo:
        base_content_dir = os.path.join(current_run_dir, "base_content")
        base_checkpoint_dir = os.path.join(base_content_dir, "checkpoints")
        miscellaneous.create_folder_if_not_exist(base_content_dir)
        os.makedirs(base_checkpoint_dir, exist_ok=True)
    os.makedirs(current_checkpoint_dir, exist_ok=True) # QUESTION: is this necessary?
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)
    
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    # wandb.login(key="d00c9f41bdf432ec2cd6df65495965d629331898")
    # wandb.init(project=prj, config=args)
    # wandb.log({'time_stamp': current_time})

    # if args.dataset in ['YahooAnswers']:
    #     read_labels = True
    # else:
    #     read_labels = False
    read_labels = True

    # load a preprocessed dataset
    if args.model == "SAE_NTM":
        dataset = datasethandler.SAEDatasetHandler(os.path.join(DATA_DIR, args.dataset), 
                                                   batch_size=args.batch_size, read_labels=read_labels,
                                                    device=args.device,)
    else:
        dataset = datasethandler.BasicDatasetHandler(
            os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
            as_tensor=True, contextual_embed=True, plm_model=args.plm_model)
        # dataset = datasethandler.BasicDatasetHandler(
        #     f'{DATA_DIR}/{args.dataset}', device=args.device, read_labels=read_labels,
        #     as_tensor=True, contextual_embed=True)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()

    if args.model == "ECRTM":
        model = ECRTM(vocab_size=dataset.vocab_size, 
                      num_topics=args.num_topics, 
                      dropout=args.dropout, 
                      pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                      train_WE=args.train_WE if args.use_pretrainWE else False, 
                      weight_loss_ECR=args.weight_ECR)
    elif args.model == "FASTOPIC":
        model = FASTOPIC(vocab_size=dataset.vocab_size, num_topics=args.num_topics)
    elif args.model == "NSTM":
        model = NSTM(vocab_size=dataset.vocab_size, num_topics=args.num_topics,
                     pretrained_WE=pretrainWE if args.use_pretrainWE else None)
    elif args.model == "CTM":
        model = CTM(vocab_size=dataset.vocab_size,
                    contextual_emb_size=dataset.contextual_embed_size,
                    num_topics=args.num_topics,
                    dropout=args.dropout)
    elif args.model == "ETM":
        model = ETM(vocab_size=dataset.vocab_size, 
                    num_topics=args.num_topics,
                    pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                    train_WE=True)
    elif args.model == "ProdLDA":
        model = ProdLDA(vocab_size=dataset.vocab_size, 
                    num_topics=args.num_topics,
                    dropout=args.dropout)
    elif args.model == "WeTe":
        model = WeTe(vocab_size=dataset.vocab_size, vocab=dataset.vocab, num_topics=args.num_topics,device=args.device)
    elif args.model == "SAE_NTM":
        model = SAE(vocab_size=dataset.vocab_size, num_topics=args.num_topics, en_units=args.hidden_dim_1, dropout=0.2, sub_embed_dim=384)
    model = model.to(args.device)



    # create a trainer
    if args.model == "FASTOPIC":
        trainer = basic_trainer.FastBasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device)
    elif args.model == "WeTe":
        trainer = basic_trainer.WeteBasicTrainer(model,epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device)
    elif args.model == "CTM":
        trainer = basic_trainer.CTMBasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)
    elif args.model == "SAE_NTM":
        trainer = basic_trainer.SAEBasicTrainer(
                                            model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size
        )
    else:
        if use_dpo and args.dpo_only_preferences:
            if not args.dpo_run_dir:
                raise RuntimeError("--dpo_run_dir is required when --dpo_only_preferences is set.")
            dpo_run_dir = args.dpo_run_dir
        else:
            dpo_run_dir = base_content_dir if use_dpo else current_run_dir
        trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device,
                                            checkpoint_dir=base_checkpoint_dir,
                                            enable_dpo=use_dpo,
                                            dpo_start_epoch=args.dpo_start_epoch,
                                            dpo_weight=args.dpo_weight,
                                            dpo_alpha=args.dpo_alpha,
                                            dpo_topic_filter=args.dpo_topic_filter,
                                            dpo_llm_model=args.dpo_llm_model,
                                            dpo_only_preferences=args.dpo_only_preferences,
                                            dpo_run_dir=dpo_run_dir,
                                            dpo_dataset=args.dataset,
                                            start_epoch=0,
                                            freeze_we_epoch=args.freeze_we_epoch
                                            )


    # train the model
    
    if args.model == "FASTOPIC":
        train_simple_embedding, train_theta = trainer.train(dataset)
    # save beta, theta and top words
        beta = trainer.save_beta(current_run_dir)
        test_theta = trainer.model.get_theta(dataset.test_contextual_embed, train_simple_embedding)
        train_theta = np.asarray(train_theta.cpu())
        test_theta = np.asarray(test_theta.cpu())
    else:
        if use_dpo and args.dpo_only_preferences:
            snapshot_path = os.path.join(
                args.dpo_run_dir,
                f"dpo_snapshot_epoch_{args.dpo_start_epoch}.pth",
            )
            if not os.path.isfile(snapshot_path):
                raise RuntimeError(f"Snapshot not found: {snapshot_path}")
            # Load snapshot tensors on CPU first; RNG CPU state must stay CPU ByteTensor.
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
            snapshot_epoch = int(snapshot.get("epoch", args.dpo_start_epoch))
            if snapshot_epoch >= args.epochs:
                raise RuntimeError(
                    f"Snapshot epoch {snapshot_epoch} >= total epochs {args.epochs}."
                )
            trainer.model.load_state_dict(snapshot["model_state_dict"])
            trainer.start_epoch = snapshot_epoch
            trainer.set_resume_state(snapshot)
            logger.info(f"[DPO] Resuming from snapshot epoch {snapshot_epoch}: {snapshot_path}")
        trainer.train(dataset)
    # save beta, theta and top words
        if use_dpo and base_content_dir and not args.dpo_only_preferences:
            final_state = copy.deepcopy(trainer.model.state_dict())
            base_snapshot_dir = args.dpo_run_dir if args.dpo_only_preferences else base_content_dir
            base_snapshot_path = os.path.join(
                base_snapshot_dir, f"dpo_snapshot_epoch_{args.dpo_start_epoch}.pth"
            )
            if os.path.isfile(base_snapshot_path):
                snapshot = torch.load(base_snapshot_path, map_location="cpu", weights_only=False)
                trainer.model.load_state_dict(snapshot["model_state_dict"])
                base_beta = trainer.save_beta(base_content_dir)
                base_train_theta, base_test_theta = trainer.save_theta(dataset, base_content_dir)
                evaluate(
                    args, trainer, dataset, base_train_theta, base_test_theta,
                    base_content_dir, logger, read_labels=True, eval_tag="BASE"
                )
            else:
                logger.info(f"Base snapshot not found: {base_snapshot_path}")
            trainer.model.load_state_dict(final_state)

        beta = trainer.save_beta(current_run_dir)
        train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)

    evaluate(args, trainer, dataset, train_theta, test_theta, current_run_dir, logger, read_labels=True, eval_tag="FINAL")
