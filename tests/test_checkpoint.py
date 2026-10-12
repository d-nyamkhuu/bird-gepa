from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState


class Adapter:
    def __init__(self):
        self.evaluations = []

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluations.append((list(batch), candidate.copy()))
        scores = [float(candidate["source"] == "improved")] * len(batch)
        return EvaluationBatch(outputs=list(batch), scores=scores, trajectories=list(batch))

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {"source": [{"feedback": "Improve the answer."}]}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return {"source": "improved"}


def test_checkpoint_resumes_candidates_and_metric_budget(tmp_path):
    adapter = Adapter()
    kwargs = {
        "seed_candidate": {"source": "seed"},
        "trainset": [1, 2],
        "valset": [3, 4],
        "adapter": adapter,
        "reflection_minibatch_size": 2,
        "run_dir": str(tmp_path / "checkpoint"),
    }
    optimize(**kwargs, max_metric_calls=2)
    assert adapter.evaluations == [([3, 4], {"source": "seed"})]
    assert GEPAState.load(kwargs["run_dir"]).total_num_evals == 2
    adapter.evaluations.clear()
    optimize(**kwargs, max_metric_calls=8)
    # GEPA 0.1.4 evaluates the seed again before loading its saved state.
    assert adapter.evaluations[0] == ([3, 4], {"source": "seed"})
    state = GEPAState.load(kwargs["run_dir"])
    assert state.total_num_evals == 8
    assert state.program_candidates == [{"source": "seed"}, {"source": "improved"}]
