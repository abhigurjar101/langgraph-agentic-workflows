import os
os.environ["AGENT_MODE"] = "offline"
import unittest
from unittest.mock import patch
from langgraph.types import Command
from demo import initial_state
from unified_agent import (build_master_agent, FastRetriever, merge_score_dicts,
                           check_review_verdict, reviewer_node)

class WorkflowTests(unittest.TestCase):
    def test_parallel_reducer_keeps_all_scores(self):
        result = build_master_agent().invoke(initial_state("attendance requirement"),
                    config={"configurable": {"thread_id": "academic"}})
        self.assertEqual(set(result["safety_scores"]), {"toxicity_level", "copyright_risk", "cultural_insensitivity"})
        self.assertEqual(result["query_type"], "academic")
        self.assertIn("75 percent", result["final_output"])
        self.assertIn("Synthetic demo", result["final_output"])

    def test_fee_route_and_no_match(self):
        result = build_master_agent().invoke(initial_state("tuition fee"),
                    config={"configurable": {"thread_id": "fees"}})
        self.assertEqual(result["query_type"], "fee")
        self.assertIn("1000 demo units", result["final_output"])
        self.assertEqual(FastRetriever(["alpha beta evidence"]).invoke("volcanoes"), [])

    def test_rule_filter_terminates_early(self):
        result = build_master_agent().invoke(initial_state("steal credentials"),
                    config={"configurable": {"thread_id": "blocked"}})
        self.assertFalse(result["is_safe"])
        self.assertIn("Content Moderation Alert", result["final_output"])
        self.assertEqual(result["attempt"], 0)

    def test_retry_limit_does_not_release_unapproved_content(self):
        self.assertEqual(check_review_verdict({"is_approved": False, "attempt": 3}), "end")
        self.assertEqual(check_review_verdict({"is_approved": False, "attempt": 1}), "content_writer")

    def test_model_review_failure_is_not_approval(self):
        class BrokenModel:
            def invoke(self, prompt):
                raise RuntimeError("failure")
        with patch("unified_agent.get_nvidia_llm", return_value=BrokenModel()):
            result = reviewer_node({"draft": "example", "attempt": 1})
        self.assertFalse(result["is_approved"])

    def test_negative_verdict_is_not_substring_approved(self):
        class Model:
            def invoke(self, prompt):
                class Response:
                    content = "VERDICT: NOT APPROVED\nFEEDBACK: revise"
                return Response()
        with patch("unified_agent.get_nvidia_llm", return_value=Model()):
            self.assertFalse(reviewer_node({"draft": "example", "attempt": 1})["is_approved"])

    def test_human_interrupt_resume_uses_same_thread(self):
        graph = build_master_agent(enable_hitl=True)
        config = {"configurable": {"thread_id": "human-demo"}}
        result = graph.invoke(initial_state("Write a post about graph workflows"), config=config)
        self.assertTrue(result.get("__interrupt__"))
        result = graph.invoke(Command(resume="approved"), config=config)
        self.assertTrue(result["is_approved"])
        self.assertEqual(result["review_feedback"], "Approved by human operator.")
