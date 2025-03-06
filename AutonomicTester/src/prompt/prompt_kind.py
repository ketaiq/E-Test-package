from enum import Enum
import re


class PromptKind(Enum):
    BUGGY = "trigger a bug"
    FIXED = "does not trigger a bug but is useful for regression testing"
    SIMILAR = "similar to existing test cases and does not trigger a bug"

    def get_prompt_filename_pattern(self, version):
        return re.compile(
            rf"^prompt_{self.name.lower()}_\d+_([-A-Za-z]+)_v{version}\.txt$"
        )

    @classmethod
    def generate_help_msg(cls) -> str:
        return "; ".join(f"{scenario.name}: {scenario.value}" for scenario in cls)
