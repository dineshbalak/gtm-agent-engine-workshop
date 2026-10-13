import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import gtm_agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Test offering",
            "min_annual_revenue": 1_000_000,
            "required_tech_stack": ["CRM"],
        }
        self.prospect_profile = {}

    def make_result(self, score, revenue_fit, tech_stack_match, segment_fit):
        return gtm_agent.ProspectScore(
            score=score,
            justification="Fit assessment.",
            rubric_breakdown=gtm_agent.RubricBreakdown(
                revenue_fit=revenue_fit,
                tech_stack_match=tech_stack_match,
                segment_fit=segment_fit,
            ),
        )

    def score(self, model_result):
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = model_result
            return gtm_agent.score_prospect.func(self.prospect_profile, self.offering)

    def test_weighted_score_uses_every_component_and_ignores_model_score(self):
        first_result = self.score(self.make_result(90, 100, 75, 100))
        second_result = self.score(self.make_result(85, 100, 75, 100))

        self.assertEqual(first_result["score"], 90.0)
        self.assertEqual(second_result["score"], 90.0)

    def test_each_component_changes_total_by_its_weight(self):
        baseline = self.score(self.make_result(0, 50, 50, 50))["score"]
        changed_scores = (
            self.score(self.make_result(0, 60, 50, 50))["score"],
            self.score(self.make_result(0, 50, 60, 50))["score"],
            self.score(self.make_result(0, 50, 50, 60))["score"],
        )

        self.assertEqual(baseline, 50.0)
        self.assertEqual(changed_scores, (54.0, 54.0, 52.0))

    def test_prompt_documents_weights_and_component_justification(self):
        prompt = gtm_agent.SCORING_PROMPT

        self.assertEqual(
            gtm_agent.RUBRIC_WEIGHTS,
            {"revenue_fit": 0.4, "tech_stack_match": 0.4, "segment_fit": 0.2},
        )
        self.assertIn("from 0 to 100", prompt)
        self.assertIn("revenue_fit × 0.4", prompt)
        self.assertIn("tech_stack_match × 0.4", prompt)
        self.assertIn("segment_fit × 0.2", prompt)
        self.assertIn("all three component scores", prompt)


if __name__ == "__main__":
    unittest.main()
