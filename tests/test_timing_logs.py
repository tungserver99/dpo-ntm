import logging
import unittest
from unittest import mock
from pathlib import Path

import torch

import basic_trainer
from utils import timing


class _DummyDataset:
    def __init__(self):
        self.train_dataloader = _DummyLoader()


class _DummyLoader:
    def __init__(self):
        self.dataset = [0]

    def __iter__(self):
        yield {"data": torch.ones(1, 2), "idx": torch.tensor([0])}


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch_data):
        loss = self.scalar * 0 + batch_data["data"].sum() * 0 + 1.0
        return {"loss": loss}


class TimingLogTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("main")
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_records = []

        class _ListHandler(logging.Handler):
            def emit(inner_self, record):
                self.log_records.append(record.getMessage())

        self.logger.addHandler(_ListHandler())

    def test_log_duration_formats_seconds(self):
        timing.log_duration(self.logger, "end_to_end", 12.3456)

        self.assertIn("[TIME] end_to_end: 12.346s", self.log_records)

    def test_log_training_duration_uses_current_time_minus_train_start(self):
        timing.log_training_duration(self.logger, train_start_time=10.0, now=16.25)

        self.assertIn("[TIME] end_to_end: 6.250s", self.log_records)

    def test_train_logs_baseline_llm_and_update_timings(self):
        timeline = iter([0.0, 4.0, 4.0, 5.5, 5.5, 11.0])

        def fake_prepare_update(trainer_self, dataset_handler, epoch, optimizer=None):
            return None

        checkpoint_dir = Path("tests/.tmp_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_file in checkpoint_dir.glob("*"):
            checkpoint_file.unlink()

        try:
            trainer = basic_trainer.BasicTrainer(
                _DummyModel(),
                epochs=2,
                batch_size=1,
                device="cpu",
                checkpoint_dir=str(checkpoint_dir),
                enable_update=True,
                update_start_epoch=1,
            )

            with mock.patch.object(basic_trainer, "perf_counter", side_effect=lambda: next(timeline)):
                with mock.patch.object(basic_trainer.BasicTrainer, "_prepare_update", new=fake_prepare_update):
                    trainer.train(_DummyDataset())
        finally:
            for checkpoint_file in checkpoint_dir.glob("*"):
                checkpoint_file.unlink()
            checkpoint_dir.rmdir()

        self.assertIn("[TIME] baseline_phase: 4.000s", self.log_records)
        self.assertIn("[TIME] llm_phase: 1.500s", self.log_records)
        self.assertIn("[TIME] update_phase: 5.500s", self.log_records)


if __name__ == "__main__":
    unittest.main()
