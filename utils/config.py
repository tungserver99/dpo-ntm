import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name', default='YahooAnswers')
    
def add_logging_argument(parser):
    parser.add_argument('--wandb_prj', type=str, default='ECRTM_TM')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, default='ECRTM', choices=['ECRTM'], help='model name')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_ECR', type=float, default=100.0)
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=False, help='Enable use_pretrainWE mode')
    parser.add_argument('--train_WE', action='store_true',
                        default=False, help='Enable train_WE mode')


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step', default='StepLR')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')
    parser.add_argument('--freeze_we_epoch', type=int, default=-1,
                        help='freeze word embeddings at this epoch (-1 to disable)')
    parser.add_argument('--enable_update', action='store_true', default=False,
                        help='enable unified update phase (DPO + contrastive)')
    parser.add_argument('--update_start_epoch', type=int, default=200,
                        help='epoch to snapshot base model and start update phase')
    parser.add_argument('--update_only', action='store_true', default=False,
                        help='resume update phase from existing snapshot/artifacts in --update_dir')
    parser.add_argument('--update_dir', type=str, default='',
                        help='directory containing reusable update artifacts and snapshot')
    parser.add_argument('--update_llm_model', type=str, default='gpt-4o',
                        help='LLM model for update artifact generation')
    parser.add_argument(
        '--dpo_topic_filter',
        type=str,
        default='cv_below_avg',
        choices=['cv_below_avg', 'cv_wikipedia_below_avg', 'llm_score_1_2', 'either', 'none'],
        help='topic filter for applying DPO loss'
    )
    parser.add_argument('--dpo_weight', type=float, default=1.0,
                        help='weight for DPO loss in update phase')
    parser.add_argument('--dpo_alpha', type=float, default=1.0,
                        help='temperature for DPO loss')
    parser.add_argument('--contrastive_weight', type=float, default=10.0,
                        help='target weight for contrastive loss in update phase')
    parser.add_argument('--contrastive_ramp_epochs', type=int, default=10,
                        help='ramp epochs after update_start_epoch for contrastive weight')
    parser.add_argument('--contrastive_topk', type=int, default=2,
                        help='top-k topics per document for pseudo labels')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='temperature for contrastive loss')
    parser.add_argument('--contrastive_queue_size', type=int, default=4096,
                        help='queue size for cross-batch contrastive candidates')
    parser.add_argument('--contrastive_doc_encoder', type=str, default='BAAI/bge-small-en-v1.5',
                        help='encoder model for raw doc/topic text embeddings')
    parser.add_argument(
        '--contrastive_loss_type',
        type=str,
        default='supcon',
        choices=['supcon', 'infonce'],
        help='contrastive objective type'
    )
    parser.add_argument(
        '--doc_embedding_source',
        type=str,
        default='rebuild_from_text',
        choices=['rebuild_from_text', 'prefer_precomputed'],
        help='source for document embeddings used in contrastive update'
    )
    parser.add_argument('--force_rebuild_doc_embeddings', action='store_true', default=False,
                        help='force rebuild document embeddings even when precomputed files exist')

def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)
    parser.add_argument('--enable_llm_eval', action='store_true', default=False,
                        help='enable LLM-based topic evaluation')


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
