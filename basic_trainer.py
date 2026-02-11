import numpy as np
import random
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy
from time import time
import json
import evaluations


class BasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, 
                 lr_scheduler=None, lr_step_size=125, log_interval=5, 
                 device="cuda", checkpoint_dir=None,
                 enable_dpo=False, dpo_start_epoch=200, dpo_weight=1.0,
                 dpo_alpha=1.0, dpo_topic_filter="cv_below_avg",
                 dpo_llm_model="gpt-4o", dpo_only_preferences=False,
                 dpo_run_dir=None, dpo_dataset=None, start_epoch=0):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self._lr_scheduler = None
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.logger = logging.getLogger('main')

        self.enable_dpo = enable_dpo
        self.dpo_start_epoch = dpo_start_epoch
        self.dpo_weight = dpo_weight
        self.dpo_alpha = dpo_alpha
        self.dpo_topic_filter = dpo_topic_filter
        self.dpo_llm_model = dpo_llm_model
        self.dpo_only_preferences = dpo_only_preferences
        self.dpo_run_dir = dpo_run_dir
        self.dpo_dataset = dpo_dataset
        self.start_epoch = start_epoch

        self._dpo_ready = False
        self._dpo_prefs = None
        self._beta_ref_logits = None
        self._resume_state = None

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            # lr_scheduler = StepLR(
            #     optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            self._lr_scheduler = self.make_lr_scheduler(optimizer)

        if self._resume_state:
            opt_state = self._resume_state.get("optimizer_state_dict")
            if opt_state:
                optimizer.load_state_dict(opt_state)
            sched_state = self._resume_state.get("lr_scheduler_state_dict")
            if self._lr_scheduler and sched_state:
                self._lr_scheduler.load_state_dict(sched_state)
            rng_state = self._resume_state.get("rng_state")
            if rng_state:
                if "torch" in rng_state:
                    torch_state = rng_state["torch"]
                    if not torch.is_tensor(torch_state):
                        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
                    torch.set_rng_state(torch_state.detach().to(device="cpu", dtype=torch.uint8))
                if "cuda" in rng_state and torch.cuda.is_available():
                    cuda_state = rng_state["cuda"]
                    if not isinstance(cuda_state, (list, tuple)):
                        cuda_state = [cuda_state]
                    normalized_cuda_state = []
                    for state in cuda_state:
                        if not torch.is_tensor(state):
                            state = torch.tensor(state, dtype=torch.uint8)
                        normalized_cuda_state.append(state.detach().to(device="cpu", dtype=torch.uint8))
                    torch.cuda.set_rng_state_all(normalized_cuda_state)
                if "numpy" in rng_state:
                    np.random.set_state(rng_state["numpy"])
                if "python" in rng_state:
                    random.setstate(rng_state["python"])

        if self.enable_dpo and not self._dpo_ready and self.start_epoch >= self.dpo_start_epoch:
            self._prepare_dpo(dataset_handler, self.dpo_start_epoch, optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(self.start_epoch + 1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            # wandb.log({'epoch': epoch})

            for batch_data in dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                if self.enable_dpo and self._dpo_ready:
                    from dpo.dpo_loss import dpo_loss
                    beta_logits = self._get_beta_logits()
                    dpo = dpo_loss(
                        beta_logits,
                        self._beta_ref_logits,
                        self._dpo_prefs,
                        alpha=self.dpo_alpha,
                        device=self.device,
                    )
                    batch_loss = batch_loss + self.dpo_weight * dpo
                    rst_dict["loss_dpo"] = dpo

                optimizer.zero_grad()
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                optimizer.step()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            # for key in loss_rst_dict:
                # wandb.log({key: loss_rst_dict[key] / data_size})
            
            if self._lr_scheduler:
                self._lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)
                self.logger.info(output_log)
            
            if epoch == self.dpo_start_epoch and self.checkpoint_dir is not None:
                self.save_checkpoint(epoch, optimizer)

            if self.enable_dpo and epoch == self.dpo_start_epoch:
                self._prepare_dpo(dataset_handler, epoch, optimizer)

        if self.enable_dpo and self._dpo_ready:
            self._log_eval_metrics(dataset_handler, eval_tag="POST_DPO")
    
    def save_checkpoint(self, epoch, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': self._lr_scheduler.state_dict() if self._lr_scheduler else None
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpint saved: {checkpoint_path}")
        
        return checkpoint_path

    def _get_beta_logits(self):
        if hasattr(self.model, "get_beta_logits"):
            return self.model.get_beta_logits()
        beta = self.model.get_beta()
        return (beta + 1e-12).log()

    def _prepare_dpo(self, dataset_handler, epoch, optimizer=None):
        if self.dpo_run_dir is None:
            self.logger.info("DPO enabled but run dir is None. Skipping DPO setup.")
            return
        if not hasattr(self.model, "get_beta"):
            self.logger.info("Model has no get_beta; skipping DPO setup.")
            return

        self._log_eval_metrics(dataset_handler, eval_tag="PRE_DPO")

        if self.dpo_only_preferences:
            # Use existing artifacts from dpo_run_dir without overwriting
            prefs_path = os.path.join(self.dpo_run_dir, "preferences.jsonl")
            if not os.path.isfile(prefs_path):
                raise RuntimeError("preferences.jsonl not found while dpo_only_preferences is set.")
            top15_path = os.path.join(self.dpo_run_dir, "top_words_15.txt")
            if not os.path.isfile(top15_path):
                raise RuntimeError("top_words_15.txt not found while dpo_only_preferences is set.")
            beta_ref_path = os.path.join(self.dpo_run_dir, "beta_ref_logits.npy")
            if not os.path.isfile(beta_ref_path):
                raise RuntimeError("beta_ref_logits.npy not found while dpo_only_preferences is set.")
            def _load_top_words_txt(path):
                lines = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        lines.append(line)
                return lines

            top_words_15 = _load_top_words_txt(top15_path)
            beta_ref_logits = np.load(beta_ref_path)
        else:
            # Save snapshot
            snapshot_path = os.path.join(self.dpo_run_dir, f"dpo_snapshot_epoch_{epoch}.pth")
            snapshot = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "lr_scheduler_state_dict": self._lr_scheduler.state_dict() if self._lr_scheduler else None,
                "rng_state": self._capture_rng_state(),
            }
            torch.save(snapshot, snapshot_path)
            self.logger.info(f"DPO snapshot saved: {snapshot_path}")

            # Save top words at E
            vocab = dataset_handler.vocab
            top_words_10 = self.save_top_words(vocab, 10, self.dpo_run_dir)
            top_words_15 = self.save_top_words(vocab, 15, self.dpo_run_dir)
            self.save_top_words(vocab, 20, self.dpo_run_dir)
            self.save_top_words(vocab, 25, self.dpo_run_dir)

            # Save beta ref logits
            beta_ref_logits = self._get_beta_logits().detach().cpu().numpy()
            beta_ref_path = os.path.join(self.dpo_run_dir, "beta_ref_logits.npy")
            np.save(beta_ref_path, beta_ref_logits)
            self.logger.info(f"beta_ref_logits saved: {beta_ref_path}")

        # Build preferences (LLM + embeddings)
        if self.dpo_only_preferences:
            prefs_path = os.path.join(self.dpo_run_dir, "preferences.jsonl")
            from dpo.jsonl_io import read_jsonl
            prefs_list = read_jsonl(prefs_path)
            prefs = {}
            for x in prefs_list:
                k = int(x["k"])
                if "w_win_indices" in x and "w_loose_indices" in x:
                    prefs[k] = {"w_win": x["w_win_indices"], "w_loose": x["w_loose_indices"]}
                else:
                    prefs[k] = {
                        "w_win": x.get("w_plus_indices", []),
                        "w_loose": x.get("w_minus_indices", []),
                    }
            pipeline = {
                "scores": {},
                "descriptions": {},
                "preferences": prefs,
            }
        else:
            from dpo.preference_builder import build_preference_pipeline
            plm_model = getattr(dataset_handler, "plm_model", "all-mpnet-base-v2")
            pipeline = build_preference_pipeline(
                run_dir=self.dpo_run_dir,
                vocab=vocab,
                plm_model=plm_model,
                llm_model=self.dpo_llm_model,
                device=self.device,
                resume=True,
                only_preferences=False,
            )

        # Topic filter
        use_topics = set()
        if self.dpo_topic_filter == "none":
            use_topics = set(pipeline["preferences"].keys())
        if self.dpo_topic_filter in ["cv_below_avg", "either"]:
            from evaluations.topic_coherence import compute_topic_coherence
            cv_list, cv_mean = compute_topic_coherence(
                dataset_handler.train_texts, dataset_handler.vocab, top_words_15, cv_type="c_v"
            )
            for k, cv in enumerate(cv_list):
                if cv < cv_mean:
                    use_topics.add(k)
        if self.dpo_topic_filter in ["llm_score_1_2", "either"]:
            scores = pipeline.get("scores", {})
            for k, s in scores.items():
                if int(s) <= 2:
                    use_topics.add(int(k))

        prefs = pipeline["preferences"]

        if self.dpo_topic_filter == "none":
            filtered_prefs = prefs
        else:
            filtered_prefs = {k: v for k, v in prefs.items() if k in use_topics}

        self._dpo_prefs = filtered_prefs
        self._beta_ref_logits = torch.from_numpy(beta_ref_logits).to(self.device)
        self._dpo_ready = True

        # Save selected topics
        topics_path = os.path.join(self.dpo_run_dir, "dpo_selected_topics.jsonl")
        with open(topics_path, "w", encoding="utf-8") as f:
            for k in sorted(use_topics):
                f.write(json.dumps({"k": k}) + "\n")
        self.logger.info(f"DPO selected topics saved: {topics_path}")

    def _log_eval_metrics(self, dataset_handler, eval_tag=None):
        if eval_tag:
            print(f"[EVAL] {eval_tag}")
            self.logger.info(f"[EVAL] {eval_tag}")

        vocab = dataset_handler.vocab
        top_words_15 = self.export_top_words(vocab, 15)

        TD_15 = evaluations.compute_topic_diversity(top_words_15, _type="TD")
        print(f"TD_15: {TD_15:.5f}")
        self.logger.info(f"TD_15: {TD_15:.5f}")

        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)

        has_labels = (
            hasattr(dataset_handler, "train_labels")
            and hasattr(dataset_handler, "test_labels")
            and dataset_handler.train_labels is not None
            and dataset_handler.test_labels is not None
        )
        if has_labels:
            clustering_results = evaluations.evaluate_clustering(
                test_theta, dataset_handler.test_labels
            )
            print(f"NMI: ", clustering_results["NMI"])
            print(f"Purity: ", clustering_results["Purity"])
            self.logger.info(f"NMI: {clustering_results['NMI']}")
            self.logger.info(f"Purity: {clustering_results['Purity']}")

            classification_results = evaluations.evaluate_classification(
                train_theta,
                test_theta,
                dataset_handler.train_labels,
                dataset_handler.test_labels,
                tune=False,
            )
            print(f"Accuracy: ", classification_results["acc"])
            print(f"Macro-f1", classification_results["macro-F1"])
            self.logger.info(f"Accuracy: {classification_results['acc']}")
            self.logger.info(f"Macro-f1: {classification_results['macro-F1']}")

        if hasattr(dataset_handler, "train_texts"):
            TC_train_list, TC_train = evaluations.compute_topic_coherence(
                dataset_handler.train_texts,
                dataset_handler.vocab,
                top_words_15,
                cv_type="c_v",
            )
            print(f"TC_train: {TC_train:.5f}")
            self.logger.info(f"TC_train: {TC_train:.5f}")
            self.logger.info(f"TC_train list: {TC_train_list}")

    def set_resume_state(self, state):
        self._resume_state = state

    def _capture_rng_state(self):
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        time1 = time()
        test_theta = self.test(dataset_handler.test_data)
        time2 = time()
        print("Time inference: " + str(time2 - time1))
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings

class FastBasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, device = "cuda"):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        
        self.train_simple_embedding = None
        self.train_theta = None

        self.logger = logging.getLogger('main')

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    # def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
    #     self.train(dataset_handler, verbose)
    #     top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
    #     train_theta = self.test(dataset_handler.train_data)

    #     return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in dataset_handler.train_dataloader:

                #rst_dict = self.model(batch_data, epoch_id=epoch)
                rst_dict = self.model(batch_data["data"], batch_data["contextual_embed"])
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                optimizer.step()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)


            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)
                self.logger.info(output_log)
                
        start_train_inference = time()      
        #self.train_simple_embedding = self.get_doc_simple_embedidng(dataset_handler.train_data)
        self.train_simple_embedding = dataset_handler.train_contextual_embed
        self.train_theta = self.model.get_theta(self.train_simple_embedding, self.train_simple_embedding)
        end_train_inference = time()
        print("*******************")
        print("Train inference time: " + str(end_train_inference - start_train_inference))
        print("*******************")
        return self.train_simple_embedding, self.train_theta
                
    def get_doc_simple_embedidng(self, input_bow):
        data_size = input_bow.shape[0]
        doc_simple_embedding = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_bow[idx]
                batch_embedding = self.model.simpleembedding(batch_input)
                doc_simple_embedding.extend(batch_embedding.cpu().tolist())

        doc_simple_embedding = torch.tensor(doc_simple_embedding).to(self.device)
        return doc_simple_embedding

    def test(self, input_data, train_simple_embeddings):
        test_simple_embedding = self.get_doc_simple_embedidng(input_data)
        data_size = test_simple_embedding.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = test_simple_embedding[idx]
                batch_theta = self.model.get_theta(batch_input, train_simple_embeddings)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        test_theta = self.test(dataset_handler.test_data, self.train_simple_embedding)
        return test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, train_theta, dataset_handler, dir_path):
        test_theta = self.export_theta(dataset_handler)
        train_theta = np.asarray(train_theta.cpu())
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings
    

class WeteBasicTrainer:
    def __init__(self, model, epochs=500, learning_rate=1e-2, batch_size=500, lr_scheduler=None, lr_step_size=125, log_interval=5, device="cuda"):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device

        self.logger = logging.getLogger('main')
    
    def to_list(self,data, device='cuda:0'):
        data_list = []
        for i in range(len(data)):
            idx = torch.where(data[i]>0)[0]
            data_list.append(torch.tensor([j for j in idx for _ in range(data[i,j])], device=device))
        return data_list

    def make_optimizer(self,):
        # args_dict = {
        #     'params': self.model.parameters(),
        #     'lr': self.learning_rate,
        # }

        # optimizer = torch.optim.Adam(**args_dict)
        self.trainable_params = []
        # print('WeTe learnable params:')
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                print(name)
                self.trainable_params.append(params)
        optimizer = torch.optim.Adam(self.trainable_params, lr=self.learning_rate, weight_decay=1e-3)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            # wandb.log({'epoch': epoch})

            for batch_data in dataset_handler.train_dataloader:
                train_data = self.to_list(batch_data["data"].long(), device=self.device)
                bow = batch_data["data"].to(self.device).float()

                rst_dict = self.model(train_data, bow )
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                for p in self.trainable_params:
                    try:
                        p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                        p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
                        torch.nn.utils.clip_grad_norm_(p, max_norm=20, norm_type=2)
                    except:
                        pass
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                optimizer.step()

                # for key in rst_dict:
                #     try:
                #         loss_rst_dict[key] += rst_dict[key] * \
                #             len(batch_data['data'])
                #     except:
                #         loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            # for key in loss_rst_dict:
                # wandb.log({key: loss_rst_dict[key] / data_size})
            
            if self.lr_scheduler:
                lr_scheduler.step()

            # if verbose and epoch % self.log_interval == 0:
            #     output_log = f'Epoch: {epoch:03d}'
            #     for key in loss_rst_dict:
            #         output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            #     print(output_log)
            #     self.logger.info(output_log)

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings


class CTMBasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.logger = logging.getLogger('main')

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            # wandb.log({'epoch': epoch})

            for batch_data in dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), True)
                optimizer.step()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            # for key in loss_rst_dict:
            #     wandb.log({key: loss_rst_dict[key] / data_size})
            
            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)
                self.logger.info(output_log)

    def test(self, input_data, input_emb):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = input_data[idx]
                batch_emb = input_emb[idx]
                batch_input = torch.cat((batch_bow, batch_emb), dim=1)
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data, dataset_handler.train_contextual_embed)
        test_theta = self.test(dataset_handler.test_data, dataset_handler.test_contextual_embed)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings


class SAEBasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200,
                 lr_scheduler=None, lr_step_size=125, log_interval=5, device="cuda"):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device

        self.logger = logging.getLogger('main')

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def test(self, input_data, sub_lists):
        data_size = input_data.shape[0]
        theta = []
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = input_data[idx]
                batch_sub_list = [sub_lists[i.item()] for i in idx]  # ✅ fix
                batch_theta = self.model.get_theta(batch_bow, sub_lists=batch_sub_list)
                if isinstance(batch_theta, tuple):  # training returns (theta, mu, logvar)
                    batch_theta = batch_theta[0]
                theta.extend(batch_theta.cpu().tolist())

        return np.asarray(theta)

    # ---------- updated fit_transform to use dataloader ----------
    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        # get theta for train set via loader (has sub_contextual_embed)
        train_theta = self.test_loader(dataset_handler.train_dataloader)
        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in dataset_handler.train_dataloader:
                # batch_data already has 'data' and 'sub_contextual_embed'
                rst_dict = self.model(batch_data)       # forward(dict) → {'loss': ...}
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # accumulate weighted by batch size (len of 'data')
                bs = batch_data['data'].shape[0]
                for key, val in rst_dict.items():
                    if torch.is_tensor(val):
                        loss_rst_dict[key] += val.item() * bs
                    else:
                        loss_rst_dict[key] += float(val) * bs

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'
                print(output_log)
                self.logger.info(output_log)

    # ---------- updated export helpers to use loaders ----------
    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data, dataset_handler.train_sub_sent_embeddings)
        time1 = time()
        test_theta = self.test(dataset_handler.test_data, dataset_handler.test_sub_sent_embeddings)
        time2 = time()
        print("Time inference: " + str(time2 - time1))
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)

        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta
