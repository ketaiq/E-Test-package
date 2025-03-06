from enum import Enum

class BuggyUnitPromptComponent(Enum):
    BuggyUnit = "function_candidate"
    ExistingTestCases = "existing_test_cases"
    NewScenario = "new_input"

class FixedUnitPromptComponent(Enum):
    FixedUnit = "function_candidate"
    ExistingTestCases = "existing_test_cases"
    NewScenario = "new_input"

class SimilarUnitPromptComponent(Enum):
    BuggyUnit = "function_candidate"
    ExistingTestCases = "existing_test_cases"
    SimilarScenario = "new_input"