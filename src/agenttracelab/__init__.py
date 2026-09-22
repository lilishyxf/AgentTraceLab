"""AgentTraceLab public package surface."""

from agenttracelab.annotation_workflow import (
    AnnotationBenchmarkBuild,
    TrajectoryManifestBuild,
    build_annotation_benchmark_manifest,
    materialize_tool_trajectory_manifest,
)
from agenttracelab.claim_evidence import ClaimEvidenceReport, verify_deepresearch_claims
from agenttracelab.curation import CuratedChallengeSet, curate_hard_cases
from agenttracelab.deepresearch_benchmark import (
    DeepResearchBenchmarkReport,
    RequiredConcept,
    TaskAcceptanceContract,
    rescore_deepresearch_benchmark,
    run_deepresearch_benchmark,
)
from agenttracelab.deepresearch_regression import (
    DeepResearchRegressionReport,
    compare_deepresearch_benchmarks,
)
from agenttracelab.evaluation import evaluate_trace
from agenttracelab.fault_injection import (
    DeepResearchFaultReport,
    run_deepresearch_fault_suite,
)
from agenttracelab.gepa_runtime import (
    GepaCandidateValidationReport,
    GepaExperimentReport,
    run_gepa_experiment,
    validate_gepa_candidate,
)
from agenttracelab.mining import HardCaseMiningReport, mine_gepa_hard_cases
from agenttracelab.models import EvaluationReport, TraceEnvelope
from agenttracelab.optimization import (
    AgentLightningRewardEventExport,
    GepaEvaluationRecord,
    OptimizationFeedback,
    TrainingRewardRecord,
    build_agent_lightning_reward_event,
    build_gepa_evaluation,
    build_optimization_feedback,
    build_training_reward,
    localize_failures,
)
from agenttracelab.readiness import (
    EvaluationReadinessReport,
    audit_deepresearch_evidence,
)
from agenttracelab.stability import GepaStabilityReport, run_gepa_stability_gate
from agenttracelab.tool_trajectory import (
    ToolTrajectoryAnnotationQueue,
    ToolTrajectoryReport,
    evaluate_tool_trajectory_manifest,
    load_tool_trajectory_annotation_queue,
)

__all__ = [
    "AgentLightningRewardEventExport",
    "AnnotationBenchmarkBuild",
    "CuratedChallengeSet",
    "ClaimEvidenceReport",
    "DeepResearchBenchmarkReport",
    "DeepResearchFaultReport",
    "DeepResearchRegressionReport",
    "EvaluationReport",
    "EvaluationReadinessReport",
    "GepaEvaluationRecord",
    "GepaCandidateValidationReport",
    "GepaExperimentReport",
    "GepaStabilityReport",
    "HardCaseMiningReport",
    "OptimizationFeedback",
    "RequiredConcept",
    "TaskAcceptanceContract",
    "TraceEnvelope",
    "TrainingRewardRecord",
    "TrajectoryManifestBuild",
    "ToolTrajectoryReport",
    "ToolTrajectoryAnnotationQueue",
    "build_agent_lightning_reward_event",
    "build_annotation_benchmark_manifest",
    "build_gepa_evaluation",
    "build_optimization_feedback",
    "build_training_reward",
    "audit_deepresearch_evidence",
    "curate_hard_cases",
    "compare_deepresearch_benchmarks",
    "evaluate_trace",
    "evaluate_tool_trajectory_manifest",
    "localize_failures",
    "materialize_tool_trajectory_manifest",
    "load_tool_trajectory_annotation_queue",
    "mine_gepa_hard_cases",
    "rescore_deepresearch_benchmark",
    "run_gepa_experiment",
    "run_deepresearch_benchmark",
    "run_deepresearch_fault_suite",
    "run_gepa_stability_gate",
    "validate_gepa_candidate",
    "verify_deepresearch_claims",
]
__version__ = "2.3.0"
