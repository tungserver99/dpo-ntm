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


class BasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200,
                 lr_scheduler=None, lr_step_size=125, log_interval=5,
                 device="cuda", checkpoint_dir=None,
                 enable_update=False, update_start_epoch=200, update_only=False,
                 update_dir=None, update_llm_model="gpt-4o", update_dataset=None,
                 dpo_weight=1.0, dpo_alpha=1.0, dpo_topic_filter="cv_below_avg",
                 contrastive_weight=10.0, contrastive_ramp_epochs=10,
                 contrastive_topk=2, contrastive_temperature=0.07,
                 contrastive_queue_size=4096,
                 contrastive_doc_encoder="BAAI/bge-small-en-v1.5",
                 contrastive_loss_type="supcon",
                 doc_embedding_source="rebuild_from_text",
                 force_rebuild_doc_embeddings=False,
                 start_epoch=0, freeze_we_epoch=-1):
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

        self.logger = logging.getLogger("main")

        self.enable_update = enable_update
        self.update_start_epoch = update_start_epoch
        self.update_only = update_only
        self.update_dir = update_dir
        self.update_llm_model = update_llm_model
        self.update_dataset = update_dataset

        self.dpo_weight = dpo_weight
        self.dpo_alpha = dpo_alpha
        self.dpo_topic_filter = dpo_topic_filter

        self.contrastive_weight = contrastive_weight
        self.contrastive_ramp_epochs = contrastive_ramp_epochs
        self.contrastive_topk = contrastive_topk
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_queue_size = contrastive_queue_size
        self.contrastive_doc_encoder = contrastive_doc_encoder
        self.contrastive_loss_type = contrastive_loss_type
        self.doc_embedding_source = doc_embedding_source
        self.force_rebuild_doc_embeddings = force_rebuild_doc_embeddings

        self.start_epoch = start_epoch
        self.freeze_we_epoch = freeze_we_epoch

        self._update_ready = False
        self._dpo_prefs = None
        self._beta_ref_logits = None
        self._resume_state = None
        self._contrastive_doc_topic = None
        self._contrastive_queue_theta = None
        self._contrastive_queue_topics = None

    def make_optimizer(self,):
        args_dict = {
            "params": self.model.parameters(),
            "lr": self.learning_rate,
        }
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5)
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

        if self.enable_update and not self._update_ready and self.start_epoch >= self.update_start_epoch:
            self._prepare_update(dataset_handler, self.update_start_epoch, optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(self.start_epoch + 1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            if self.freeze_we_epoch > 0 and epoch == self.freeze_we_epoch:
                if hasattr(self.model, "word_embeddings"):
                    self.model.word_embeddings.requires_grad = False
                    self.logger.info(f"Frozen word_embeddings at epoch {epoch}")

            for batch_data in dataset_handler.train_dataloader:
                rst_dict = self.model(batch_data)
                batch_loss = rst_dict["loss"]

                if self.enable_update and self._update_ready:
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

                    if self._contrastive_doc_topic is not None:
                        batch_idx = batch_data.get("idx")
                        if batch_idx is not None:
                            batch_idx = batch_idx.to(self.device, dtype=torch.long)
                            theta_batch = rst_dict.get("theta")
                            if theta_batch is None:
                                theta_batch, _ = self.model.encode(batch_data["data"])
                            topic_batch = self._contrastive_doc_topic[batch_idx].float()
                            con_loss, valid_anchors = self._compute_contrastive_loss(theta_batch, topic_batch)
                            eff_weight = self._contrastive_effective_weight(epoch)
                            if valid_anchors > 0 and eff_weight > 0:
                                batch_loss = batch_loss + eff_weight * con_loss
                                rst_dict["loss_contrastive"] = con_loss
                                rst_dict["contrastive_weight_eff"] = torch.tensor(eff_weight, device=self.device)
                            self._update_contrastive_queue(theta_batch.detach(), topic_batch.detach().bool())

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                bs = batch_data["data"].shape[0] if isinstance(batch_data, dict) and "data" in batch_data else len(batch_data)
                for key, val in rst_dict.items():
                    if torch.is_tensor(val):
                        if val.ndim != 0:
                            continue
                        loss_rst_dict[key] += val.detach().item() * bs
                    else:
                        try:
                            loss_rst_dict[key] += float(val) * bs
                        except (TypeError, ValueError):
                            continue

            if self._lr_scheduler:
                self._lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f"Epoch: {epoch:03d}"
                for key in loss_rst_dict:
                    output_log += f" {key}: {loss_rst_dict[key] / data_size :.3f}"
                print(output_log)
                self.logger.info(output_log)

            if epoch == self.update_start_epoch and self.checkpoint_dir is not None:
                self.save_checkpoint(epoch, optimizer)

            if self.enable_update and not self._update_ready and epoch == self.update_start_epoch:
                self._prepare_update(dataset_handler, epoch, optimizer)

    def save_checkpoint(self, epoch, optimizer):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": self._lr_scheduler.state_dict() if self._lr_scheduler else None,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpint saved: {checkpoint_path}")
        return checkpoint_path

    def _get_beta_logits(self):
        if hasattr(self.model, "get_beta_logits"):
            return self.model.get_beta_logits()
        beta = self.model.get_beta()
        return (beta + 1e-12).log()

    def _prepare_update(self, dataset_handler, epoch, optimizer=None):
        if self.update_dir is None:
            self.logger.info("Update enabled but update_dir is None. Skipping update setup.")
            return
        if not hasattr(self.model, "get_beta"):
            self.logger.info("Model has no get_beta; skipping update setup.")
            return
        if hasattr(self.model, "word_embeddings"):
            self.model.word_embeddings.requires_grad = False

        def _load_top_words_txt(path):
            lines = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
            return lines

        from dpo.jsonl_io import read_jsonl

        if self.update_only:
            prefs_path = os.path.join(self.update_dir, "preferences.jsonl")
            if not os.path.isfile(prefs_path):
                raise RuntimeError("preferences.jsonl not found while update_only is set.")
            top15_path = os.path.join(self.update_dir, "top_words_15.txt")
            if not os.path.isfile(top15_path):
                raise RuntimeError("top_words_15.txt not found while update_only is set.")
            beta_ref_path = os.path.join(self.update_dir, "beta_ref_logits.npy")
            if not os.path.isfile(beta_ref_path):
                raise RuntimeError("beta_ref_logits.npy not found while update_only is set.")

            top_words_15 = _load_top_words_txt(top15_path)
            beta_ref_logits = np.load(beta_ref_path)
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
            scores_path = os.path.join(self.update_dir, "topic_scores.jsonl")
            desc_path = os.path.join(self.update_dir, "topic_descriptions.jsonl")
            scores = {int(x["k"]): int(x["llm_score"]) for x in read_jsonl(scores_path)} if os.path.isfile(scores_path) else {}
            descriptions = {int(x["k"]): str(x.get("main_meaning", "")) for x in read_jsonl(desc_path)} if os.path.isfile(desc_path) else {}
            pipeline = {
                "scores": scores,
                "descriptions": descriptions,
                "preferences": prefs,
            }
        else:
            snapshot_path = os.path.join(self.update_dir, f"update_snapshot_epoch_{epoch}.pth")
            snapshot = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "lr_scheduler_state_dict": self._lr_scheduler.state_dict() if self._lr_scheduler else None,
                "rng_state": self._capture_rng_state(),
            }
            torch.save(snapshot, snapshot_path)
            self.logger.info(f"Update snapshot saved: {snapshot_path}")

            vocab = dataset_handler.vocab
            self.save_top_words(vocab, 10, self.update_dir)
            self.save_top_words(vocab, 15, self.update_dir)
            self.save_top_words(vocab, 20, self.update_dir)
            self.save_top_words(vocab, 25, self.update_dir)
            top_words_15 = _load_top_words_txt(os.path.join(self.update_dir, "top_words_15.txt"))

            beta_ref_logits = self._get_beta_logits().detach().cpu().numpy()
            beta_ref_path = os.path.join(self.update_dir, "beta_ref_logits.npy")
            np.save(beta_ref_path, beta_ref_logits)
            self.logger.info(f"beta_ref_logits saved: {beta_ref_path}")

            from dpo.preference_builder import build_preference_pipeline
            plm_model = self.contrastive_doc_encoder
            pipeline = build_preference_pipeline(
                run_dir=self.update_dir,
                vocab=vocab,
                plm_model=plm_model,
                llm_model=self.update_llm_model,
                device=self.device,
                resume=True,
                only_preferences=False,
            )

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
        if self.dpo_topic_filter == "cv_wikipedia_below_avg":
            from evaluations.topic_coherence import TC_on_wikipedia_llm_itl
            top15_path = os.path.join(self.update_dir, "top_words_15.txt")
            tc_scores, tc_mean = TC_on_wikipedia_llm_itl(top15_path, tc_metric="C_V")
            for k, tc in enumerate(tc_scores):
                if tc < tc_mean:
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
        self._prepare_doc_topic_contrastive(dataset_handler, pipeline.get("descriptions", {}))
        self._update_ready = True

        topics_path = os.path.join(self.update_dir, "update_selected_topics.jsonl")
        with open(topics_path, "w", encoding="utf-8") as f:
            for k in sorted(use_topics):
                f.write(json.dumps({"k": k}) + "\n")
        self.logger.info(f"Update selected topics saved: {topics_path}")

    def _prepare_doc_topic_contrastive(self, dataset_handler, descriptions):
        if self.update_dir is None:
            return

        dataset_dir = getattr(dataset_handler, "dataset_dir", None)
        if dataset_dir is None:
            dataset_dir = os.path.join("datasets", str(self.update_dataset))

        from dpo.contrastive_doctopic import build_doc_topic_multi_hot
        num_topics = int(getattr(self.model, "num_topics"))
        multi_hot = build_doc_topic_multi_hot(
            run_dir=self.update_dir,
            dataset_dir=dataset_dir,
            num_topics=num_topics,
            topk=self.contrastive_topk,
            model_name=self.contrastive_doc_encoder,
            device=self.device,
            fallback_descriptions=descriptions,
            fallback_train_texts=getattr(dataset_handler, "train_texts", None),
            read_only=self.update_only,
            force_rebuild_embeddings=self.force_rebuild_doc_embeddings,
            doc_embedding_source=self.doc_embedding_source,
            logger=self.logger,
        )
        if multi_hot.shape[0] != len(dataset_handler.train_dataloader.dataset):
            raise RuntimeError(
                f"Contrastive doc-topic size mismatch: {multi_hot.shape[0]} vs {len(dataset_handler.train_dataloader.dataset)}"
            )
        self._contrastive_doc_topic = torch.from_numpy(multi_hot).to(self.device, dtype=torch.bool)
        self._contrastive_queue_theta = None
        self._contrastive_queue_topics = None

    def _compute_contrastive_loss(self, theta_batch, topic_batch_float):
        theta_prob = theta_batch.clamp_min(1e-12)
        if self.contrastive_loss_type == "supcon":
            from dpo.contrastive_doctopic import supcon_theta_loss
            return supcon_theta_loss(
                theta=theta_prob,
                topic_multi_hot=topic_batch_float.bool(),
                queue_theta=self._contrastive_queue_theta,
                queue_topic_multi_hot=self._contrastive_queue_topics,
                temperature=self.contrastive_temperature,
            )
        if self.contrastive_loss_type == "infonce":
            from dpo.contrastive_doctopic import infonce_theta_loss
            return infonce_theta_loss(
                theta=theta_prob,
                topic_multi_hot=topic_batch_float.bool(),
                queue_theta=self._contrastive_queue_theta,
                queue_topic_multi_hot=self._contrastive_queue_topics,
                temperature=self.contrastive_temperature,
            )
        raise RuntimeError(f"Unsupported contrastive_loss_type: {self.contrastive_loss_type}")

    def _contrastive_effective_weight(self, epoch):
        target = float(self.contrastive_weight)
        ramp = int(self.contrastive_ramp_epochs)
        if ramp <= 0:
            return target
        if epoch <= self.update_start_epoch:
            return 0.0
        progress = min(1.0, max(0.0, (epoch - self.update_start_epoch) / float(ramp)))
        return target * progress

    def _update_contrastive_queue(self, theta_detached, topic_detached):
        if self.contrastive_queue_size <= 0:
            return
        theta_cpu = theta_detached
        topic_cpu = topic_detached
        if self._contrastive_queue_theta is None:
            self._contrastive_queue_theta = theta_cpu
            self._contrastive_queue_topics = topic_cpu
        else:
            self._contrastive_queue_theta = torch.cat([self._contrastive_queue_theta, theta_cpu], dim=0)
            self._contrastive_queue_topics = torch.cat([self._contrastive_queue_topics, topic_cpu], dim=0)
        if self._contrastive_queue_theta.shape[0] > self.contrastive_queue_size:
            keep = self.contrastive_queue_size
            self._contrastive_queue_theta = self._contrastive_queue_theta[-keep:]
            self._contrastive_queue_topics = self._contrastive_queue_topics[-keep:]

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
        np.save(os.path.join(dir_path, "beta.npy"), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f"top_words_{num_top_words}.txt"), "w") as f:
            for words in top_words:
                f.write(words + "\n")
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, "train_theta.npy"), train_theta)
        np.save(os.path.join(dir_path, "test_theta.npy"), test_theta)

        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, "train_argmax_theta.npy"), train_argmax_theta)
        np.save(os.path.join(dir_path, "test_argmax_theta.npy"), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, "word_embeddings"):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "word_embeddings.npy"), word_embeddings)
            self.logger.info(f"word_embeddings size: {word_embeddings.shape}")

        if hasattr(self.model, "topic_embeddings"):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "topic_embeddings.npy"), topic_embeddings)
            self.logger.info(f"topic_embeddings size: {topic_embeddings.shape}")

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, "topic_dist.npy"), topic_dist)

        if hasattr(self.model, "group_embeddings"):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, "group_embeddings.npy"), group_embeddings)
            self.logger.info(f"group_embeddings size: {group_embeddings.shape}")

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, "group_dist.npy"), group_dist)

        return word_embeddings, topic_embeddings
