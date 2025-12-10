import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class Llama31Detector(BaseFormatDetector):
    """
    Detector for Llama 3.1 models with function call format.

    Format Structure:
    ```
    <function=function_name>{"arg1": "value1"}</function>
    ```

    Multiple function calls are separated by newlines.
    """

    def __init__(self):
        super().__init__()
        self.function_regex = re.compile(
            r"<function=([^>]+)>(.*?)</function>", re.DOTALL
        )
        self.bot_token = "<function="

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.1 format tool call."""
        return "<function=" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        all_actions = []
        normal_parts = []
        last_end = 0

        for match in self.function_regex.finditer(text):
            # Capture text before this match as normal text
            normal_parts.append(text[last_end : match.start()])
            last_end = match.end()

            func_name = match.group(1).strip()
            args_str = match.group(2).strip()

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse arguments for {func_name}: {args_str}")
                args = {}

            all_actions.append({"name": func_name, "arguments": args})

        # Capture any remaining text after the last match
        normal_parts.append(text[last_end:])
        normal_text = "".join(normal_parts).strip()

        calls = self.parse_base_json(all_actions, tools) if all_actions else []
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<function={name}>",
            end="</function>",
            trigger="<function=",
        )
