import sys
import types


sentence_transformers_stub = types.ModuleType("sentence_transformers")


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


sentence_transformers_stub.SentenceTransformer = _DummySentenceTransformer
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

openai_stub = types.ModuleType("openai")


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass


openai_stub.OpenAI = _DummyOpenAI
sys.modules.setdefault("openai", openai_stub)

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_stub)

from dpo.preference_builder import _parse_preference_entry


def test_parse_preference_entry_prefers_w_lose_indices():
    parsed = _parse_preference_entry(
        {
            "k": 3,
            "w_win_indices": [1, 2],
            "w_lose_indices": [4, 5],
        }
    )

    assert parsed == (3, {"w_win": [1, 2], "w_lose": [4, 5]})


def test_parse_preference_entry_supports_legacy_w_loose_indices():
    parsed = _parse_preference_entry(
        {
            "k": 7,
            "w_win_indices": [10],
            "w_loose_indices": [11, 12],
        }
    )

    assert parsed == (7, {"w_win": [10], "w_lose": [11, 12]})
