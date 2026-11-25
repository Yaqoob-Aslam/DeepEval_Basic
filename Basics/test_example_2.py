from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepSeekModel

# Load DEEPSEEK_API_KEY from the project's .env file
load_dotenv()


def test_answer_relevancy():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        actual_output="Paris.",
    )

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=DeepSeekModel(model="deepseek-chat"),
    )

    assert_test(test_case, [metric])
