from utils import config, log, miscellaneous, seed
import os
import basic_trainer
from models.ECRTM.ECRTM import ECRTM
import datasethandler
import scipy
from evaluate import evaluate
import torch
import copy

RESULT_DIR = "results"
DATA_DIR = "datasets"


if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(
        RESULT_DIR, str(args.model), str(args.dataset), str(args.num_topics),
        str(args.weight_ECR) + "-" + str(args.epochs) + "-" + current_time,
    )
    current_checkpoint_dir = os.path.join(current_run_dir, "checkpoints")

    use_update = args.enable_update
    base_content_dir = None
    base_checkpoint_dir = current_checkpoint_dir
    if use_update:
        base_content_dir = os.path.join(current_run_dir, "base_content")
        base_checkpoint_dir = os.path.join(base_content_dir, "checkpoints")
        miscellaneous.create_folder_if_not_exist(base_content_dir)
        os.makedirs(base_checkpoint_dir, exist_ok=True)

    os.makedirs(current_checkpoint_dir, exist_ok=True)
    miscellaneous.create_folder_if_not_exist(current_run_dir)
    config.save_config(args, os.path.join(current_run_dir, "config.txt"))

    seed.seedEverything(args.seed)
    print(args)
    logger = log.setup_logger("main", os.path.join(current_run_dir, "main.log"))

    read_labels = True
    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset),
        device=args.device,
        read_labels=read_labels,
        as_tensor=True,
        contextual_embed=True,
        plm_model=args.plm_model,
    )

    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz"
    )).toarray()
    model = ECRTM(
        vocab_size=dataset.vocab_size,
        num_topics=args.num_topics,
        dropout=args.dropout,
        pretrained_WE=pretrainWE if args.use_pretrainWE else None,
        train_WE=args.train_WE if args.use_pretrainWE else False,
        weight_loss_ECR=args.weight_ECR,
    ).to(args.device)

    if use_update and args.update_only:
        if not args.update_dir:
            raise RuntimeError("--update_dir is required when --update_only is set.")
        update_dir = args.update_dir
    else:
        update_dir = base_content_dir if use_update else current_run_dir

    trainer = basic_trainer.BasicTrainer(
        model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        device=args.device,
        checkpoint_dir=base_checkpoint_dir,
        enable_update=use_update,
        update_start_epoch=args.update_start_epoch,
        update_only=args.update_only,
        update_dir=update_dir,
        update_llm_model=args.update_llm_model,
        update_dataset=args.dataset,
        dpo_weight=args.dpo_weight,
        dpo_alpha=args.dpo_alpha,
        dpo_topic_filter=args.dpo_topic_filter,
        contrastive_weight=args.contrastive_weight,
        contrastive_ramp_epochs=args.contrastive_ramp_epochs,
        contrastive_topk=args.contrastive_topk,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_queue_size=args.contrastive_queue_size,
        contrastive_doc_encoder=args.contrastive_doc_encoder,
        contrastive_loss_type=args.contrastive_loss_type,
        doc_embedding_source=args.doc_embedding_source,
        force_rebuild_doc_embeddings=args.force_rebuild_doc_embeddings,
        start_epoch=0,
        freeze_we_epoch=args.freeze_we_epoch,
    )

    if use_update and args.update_only:
        snapshot_path = os.path.join(
            args.update_dir,
            f"update_snapshot_epoch_{args.update_start_epoch}.pth",
        )
        if not os.path.isfile(snapshot_path):
            raise RuntimeError(f"Snapshot not found: {snapshot_path}")
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        snapshot_epoch = int(snapshot.get("epoch", args.update_start_epoch))
        if snapshot_epoch >= args.epochs:
            raise RuntimeError(
                f"Snapshot epoch {snapshot_epoch} >= total epochs {args.epochs}."
            )
        trainer.model.load_state_dict(snapshot["model_state_dict"])
        trainer.start_epoch = snapshot_epoch
        trainer.set_resume_state(snapshot)
        logger.info(f"[UPDATE] Resuming from snapshot epoch {snapshot_epoch}: {snapshot_path}")

    trainer.train(dataset)

    if use_update and base_content_dir and not args.update_only:
        final_state = copy.deepcopy(trainer.model.state_dict())
        base_snapshot_path = os.path.join(
            base_content_dir, f"update_snapshot_epoch_{args.update_start_epoch}.pth"
        )
        if os.path.isfile(base_snapshot_path):
            snapshot = torch.load(base_snapshot_path, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(snapshot["model_state_dict"])
            base_train_theta, base_test_theta = trainer.save_theta(dataset, base_content_dir)
            evaluate(
                args, trainer, dataset, base_train_theta, base_test_theta,
                base_content_dir, logger, read_labels=True, eval_tag="BASE"
            )
        else:
            logger.info(f"Base snapshot not found: {base_snapshot_path}")
        trainer.model.load_state_dict(final_state)

    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    final_tag = "UPDATED" if use_update else "BASE"
    evaluate(
        args, trainer, dataset, train_theta, test_theta,
        current_run_dir, logger, read_labels=True, eval_tag=final_tag
    )
