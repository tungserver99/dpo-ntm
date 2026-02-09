import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name', default='YahooAnswers')
    parser.add_argument('--plm_model', type=str,
                        help='plm model name', default='all-mpnet-base-v2')
    
def add_logging_argument(parser):
    parser.add_argument('--wandb_prj', type=str, default='SAE_TM')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, default='ZTM', help='model name')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_groups', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dim_1', type=int, default=384)
    parser.add_argument('--hidden_dim_2', type=int, default=384)
    parser.add_argument('--theta_temp', type=float, default=1.0)
    parser.add_argument('--DT_alpha', type=float, default=3.0)
    parser.add_argument('--TW_alpha', type=float, default=2.0)
    
    parser.add_argument('--weight_GR', type=float, default=1.)
    parser.add_argument('--alpha_GR', type=float, default=5.)
    parser.add_argument('--weight_InfoNCE', type=float, default=50.)
    parser.add_argument('--beta_temp', type=float, default=0.2)
    parser.add_argument('--weight_ECR', type=float, default=100.0)
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=False, help='Enable use_pretrainWE mode')
    parser.add_argument('--train_WE', action='store_true',
                        default=False, help='Enable train_WE mode')

def add_wete_argument(parser):
    parser.add_argument('--glove', type=str, default='glove.6B.100d.txt', help='embedding model name')
    parser.add_argument('--wete_beta', type=float, default=0.5)
    parser.add_argument('--wete_epsilon', type=float, default=0.1)
    parser.add_argument('--init_alpha', type=bool, default=True)


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
    parser.add_argument('--checkpoint_epoch', type=int, default=400,
                        help='checkpoint epoch to save the model')
    parser.add_argument(
        '--dpo_topic_filter',
        type=str,
        default='cv_below_avg',
        choices=['cv_below_avg', 'llm_score_1_2', 'either'],
        help='topic filter for applying DPO loss'
    )
    parser.add_argument('--enable_dpo', action='store_true', default=False,
                        help='enable DPO preference learning')
    parser.add_argument('--dpo_start_epoch', type=int, default=200,
                        help='epoch to snapshot and start DPO preference learning')
    parser.add_argument('--dpo_weight', type=float, default=1.0,
                        help='weight for DPO loss')
    parser.add_argument('--dpo_alpha', type=float, default=1.0,
                        help='temperature for DPO loss')
    parser.add_argument('--dpo_llm_model', type=str, default='gpt-4o',
                        help='LLM model for preference generation')
    parser.add_argument('--dpo_only_preferences', action='store_true', default=False,
                        help='skip preference building; use existing preferences.jsonl')
    parser.add_argument('--dpo_run_dir', type=str, default='',
                        help='override run dir for DPO artifacts (topic_scores, preferences, beta_ref_logits)')

def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)


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
