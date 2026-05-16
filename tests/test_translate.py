import os
import sys
import unittest

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.translate import (
    extract_text,
    last_user_text,
    translate_messages,
    translate_tool_choice,
    translate_tools,
)


class TestExtractText(unittest.TestCase):
    def test_string(self):
        self.assertEqual(extract_text("hello"), "hello")

    def test_non_dict_without_text(self):
        self.assertEqual(extract_text({"a": 1}), "")
        self.assertEqual(extract_text(123), "")

    def test_content_array(self):
        self.assertEqual(
            extract_text([
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"},
            ]),
            "ab",
        )

    def test_ignore_non_text(self):
        self.assertEqual(
            extract_text([
                {"type": "input_image"},
                {"type": "input_text", "text": "t"},
            ]),
            "t",
        )

    def test_single_object(self):
        self.assertEqual(extract_text({"type": "text", "text": "ok"}), "ok")


class TestTranslateMessages(unittest.TestCase):
    def test_string_input(self):
        r = translate_messages("hello")
        self.assertEqual(len(r["messages"]), 1)
        self.assertEqual(r["messages"][0]["role"], "user")

    def test_empty_string(self):
        self.assertEqual(len(translate_messages("   ")["messages"]), 0)

    def test_message_items(self):
        r = translate_messages([
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi!"}]},
        ])
        self.assertEqual(len(r["messages"]), 2)
        self.assertEqual(r["messages"][0]["content"], "hi")

    def test_developer_to_system(self):
        self.assertEqual(
            translate_messages([{"role": "developer", "content": "sys"}])["messages"][
                0
            ]["role"],
            "system",
        )

    def test_function_call_merge(self):
        r = translate_messages([
            {"role": "assistant", "content": []},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        ])
        self.assertEqual(len(r["messages"]), 1)
        self.assertEqual(len(r["messages"][0]["tool_calls"]), 2)

    def test_function_call_output(self):
        r = translate_messages([
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": {"type": "text", "text": "ok"},
            }
        ])
        self.assertEqual(r["messages"][0]["role"], "tool")
        self.assertEqual(r["messages"][0]["content"], "ok")

    def test_reasoning_skipped(self):
        r = translate_messages([
            {"role": "user", "content": "q"},
            {"type": "reasoning", "reasoning_content": "t"},
        ])
        self.assertEqual(len(r["messages"]), 1)
        self.assertEqual(r["stats"]["skipped"]["reasoning"], 1)

    def test_rc_stripped_default(self):
        r = translate_messages([
            {"role": "assistant", "content": "a", "reasoning_content": "t"}
        ])
        self.assertNotIn("reasoning_content", r["messages"][0])
        self.assertEqual(r["stats"]["strippedReasoningContent"], 1)

    def test_rc_kept(self):
        r = translate_messages(
            [{"role": "assistant", "content": "a", "reasoning_content": "t"}],
            {"keepReasoningContent": True},
        )
        self.assertEqual(r["messages"][0]["reasoning_content"], "t")
        self.assertEqual(r["stats"]["preservedReasoningContent"], 1)
        self.assertEqual(r["stats"]["strippedReasoningContent"], 0)

    def test_file_audio_stats(self):
        r = translate_messages([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hi"},
                    {"type": "input_file"},
                    {"type": "input_audio"},
                ],
            }
        ])
        self.assertEqual(r["stats"]["skipped"]["file"], 1)
        self.assertEqual(r["stats"]["skipped"]["audio"], 1)

    def test_function_call_status_incomplete(self):
        r = translate_messages([
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
                "status": "incomplete",
            }
        ])
        self.assertEqual(len(r["messages"]), 1)
        self.assertEqual(r["messages"][0]["role"], "assistant")

    def test_empty_array(self):
        self.assertEqual(len(translate_messages([])["messages"]), 0)

    def test_none(self):
        self.assertEqual(len(translate_messages(None)["messages"]), 0)

    def test_full_conversation(self):
        r = translate_messages([
            {
                "role": "user",
                "id": "1",
                "content": [{"type": "input_text", "text": "w?"}],
            },
            {
                "id": "2",
                "type": "function_call",
                "call_id": "abc",
                "name": "f",
                "arguments": "{}",
                "status": "completed",
            },
            {
                "id": "3",
                "type": "function_call_output",
                "call_id": "abc",
                "output": {"type": "text", "text": "ok"},
                "status": "completed",
            },
            {
                "role": "assistant",
                "id": "4",
                "content": [{"type": "output_text", "text": "ok!"}],
            },
        ])
        self.assertEqual(len(r["messages"]), 4)
        self.assertEqual(r["messages"][0]["role"], "user")
        self.assertEqual(r["messages"][1]["role"], "assistant")
        self.assertEqual(r["messages"][2]["role"], "tool")
        self.assertEqual(r["messages"][3]["role"], "assistant")


class TestTranslateTools(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(translate_tools(None), [])
        self.assertEqual(translate_tools([]), [])

    def test_normal(self):
        r = translate_tools([
            {"type": "function", "name": "s", "description": "d", "parameters": {}}
        ])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["function"]["name"], "s")

    def test_function_wrapper(self):
        r = translate_tools([{"function": {"name": "c", "description": "d"}}])
        self.assertEqual(r[0]["function"]["name"], "c")

    def test_filter_nameless(self):
        r = translate_tools([{"type": "function"}, {"type": "function", "name": "v"}])
        self.assertEqual(len(r), 1)


class TestTranslateToolChoice(unittest.TestCase):
    def test_null(self):
        self.assertIsNone(translate_tool_choice(None))

    def test_string_passthrough(self):
        self.assertEqual(translate_tool_choice("auto"), "auto")

    def test_object(self):
        r = translate_tool_choice({"type": "function", "name": "c"})
        self.assertEqual(r["function"]["name"], "c")  # type: ignore


class TestLastUserText(unittest.TestCase):
    def test_found(self):
        self.assertEqual(
            last_user_text([
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
            ]),
            "q2",
        )

    def test_not_found(self):
        self.assertEqual(last_user_text([]), "")


if __name__ == "__main__":
    unittest.main()
