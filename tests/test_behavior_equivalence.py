from evidenceloop import Pipeline, Document
from tests.reference.pipeline_v1 import PipelineV1
from tests.test_action_flow import (SOURCE, MockProposer, MockCritic, MockIntegrator,
    EmptyProposer, AdditionOnlyIntegrator, MissingFinalCritic, InvalidInitialPredicateCritic)
import pytest


def deterministic(record):
    value = record.to_dict()
    value.pop("stage_timings_seconds")
    return value


@pytest.mark.parametrize("proposer,critic,integrator,variant", [
    (MockProposer, MockCritic, MockIntegrator, "full"),
    (MockProposer, InvalidInitialPredicateCritic, MockIntegrator, "full"),
    (MockProposer, MockCritic, MockIntegrator, "proposer_critic_once"),
    (EmptyProposer, MissingFinalCritic, AdditionOnlyIntegrator, "full"),
    (MockProposer, MissingFinalCritic, MockIntegrator, "full"),
])
def test_original_and_refactored_document_algorithms_agree(proposer, critic, integrator, variant):
    old = PipelineV1(proposer(), critic(), integrator(), variant)
    new = Pipeline(proposer(), critic(), integrator(), pipeline_variant=variant)
    expected = deterministic(old.process_document("equivalence", SOURCE))
    assert deterministic(new.process(Document("equivalence", SOURCE))) == expected
