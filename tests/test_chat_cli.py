import unittest
from unittest import mock

from starter_kit import chat_cli


class ScriptedIO:
    """Feeds scripted inputs to run() and records everything printed, so the
    interactive loop can be tested without a real terminal."""

    def __init__(self, inputs):
        self._inputs = iter(inputs)
        self.output_lines = []

    def input_fn(self, prompt=""):
        try:
            return next(self._inputs)
        except StopIteration:
            raise EOFError

    def output(self, *args):
        self.output_lines.append(" ".join(str(a) for a in args))

    @property
    def full_output(self):
        return "\n".join(self.output_lines)


class ExitCommandTests(unittest.TestCase):
    def test_recognizes_variants(self):
        for text in ("退出", "exit", "EXIT", "quit", "q", "  退出  "):
            self.assertTrue(chat_cli._is_exit_command(text), text)

    def test_normal_question_is_not_an_exit_command(self):
        self.assertFalse(chat_cli._is_exit_command("生成一个贝尔态"))


class RunLoopTests(unittest.TestCase):
    def test_normal_turn_calls_send_message_and_prints_reply(self):
        io = ScriptedIO(["生成一个贝尔态", "退出"])
        fake_session = object()
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=fake_session):
            with mock.patch.object(
                chat_cli.l2_agent, "send_message", return_value="这是电路..."
            ) as mocked:
                code = chat_cli.run(input_fn=io.input_fn, output=io.output)
        mocked.assert_called_once_with(fake_session, "生成一个贝尔态")
        self.assertEqual(code, 0)
        self.assertIn("这是电路...", io.full_output)

    def test_empty_input_is_skipped_without_calling_send_message(self):
        io = ScriptedIO(["", "   ", "退出"])
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(chat_cli.l2_agent, "send_message") as mocked:
                chat_cli.run(input_fn=io.input_fn, output=io.output)
        mocked.assert_not_called()

    def test_eof_ends_the_session_cleanly(self):
        io = ScriptedIO([])  # first input_fn call raises EOFError
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(chat_cli.l2_agent, "send_message") as mocked:
                code = chat_cli.run(input_fn=io.input_fn, output=io.output)
        mocked.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("再见", io.full_output)

    def test_keyboard_interrupt_ends_the_session_cleanly(self):
        def raising_input_fn(prompt=""):
            raise KeyboardInterrupt

        io = ScriptedIO([])
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            code = chat_cli.run(input_fn=raising_input_fn, output=io.output)
        self.assertEqual(code, 0)
        self.assertIn("再见", io.full_output)

    def test_runtime_error_shows_actionable_hint_and_continues(self):
        io = ScriptedIO(["生成一个贝尔态", "退出"])
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(
                chat_cli.l2_agent,
                "send_message",
                side_effect=RuntimeError(
                    "missing required LoomQ L2 environment variable(s): LOOMQ_LLM_BASE_URL"
                ),
            ):
                code = chat_cli.run(input_fn=io.input_fn, output=io.output)
        self.assertEqual(code, 0)  # reached the "退出" turn, didn't crash
        self.assertIn("LOOMQ_LLM_BASE_URL", io.full_output)
        self.assertIn("环境变量", io.full_output)

    def test_unexpected_exception_does_not_crash_the_session(self):
        io = ScriptedIO(["生成一个贝尔态", "退出"])
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(chat_cli.l2_agent, "send_message", side_effect=ValueError("boom")):
                code = chat_cli.run(input_fn=io.input_fn, output=io.output)
        self.assertEqual(code, 0)
        self.assertIn("boom", io.full_output)

    def test_visualization_is_printed_after_a_reply_containing_qasm(self):
        io = ScriptedIO(["生成一个贝尔态", "退出"])
        reply_with_qasm = (
            "[TASK: generate]\n[TARGET: ghz]\n```\nOPENQASM 2.0;\ninclude "
            '"qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n'
            "measure q -> c;\n```\n"
        )
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(chat_cli.l2_agent, "send_message", return_value=reply_with_qasm):
                with mock.patch.object(
                    chat_cli, "_visualize_circuit_result", return_value="实际运行了 1024 次..."
                ) as mocked_viz:
                    chat_cli.run(input_fn=io.input_fn, output=io.output)
        mocked_viz.assert_called_once_with(reply_with_qasm)
        self.assertIn("实际运行了 1024 次...", io.full_output)

    def test_no_visualization_line_when_reply_has_no_qasm(self):
        io = ScriptedIO(["哪个后端比较好？", "退出"])
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=object()):
            with mock.patch.object(
                chat_cli.l2_agent, "send_message", return_value="[TASK: select_backend]\n用 xxx 吧"
            ):
                chat_cli.run(input_fn=io.input_fn, output=io.output)
        self.assertNotIn("实际运行了", io.full_output)

    def test_same_session_object_is_reused_across_turns(self):
        # This is the actual memory guarantee: every call to send_message
        # within one CLI run must receive the SAME session object, not a
        # fresh one per turn, or there is no cross-turn memory at all.
        io = ScriptedIO(["生成一个贝尔态", "把比特数改成 5 个", "退出"])
        fake_session = object()
        with mock.patch.object(chat_cli.l2_agent, "start_session", return_value=fake_session) as start_mock:
            with mock.patch.object(chat_cli.l2_agent, "send_message", return_value="ok") as send_mock:
                chat_cli.run(input_fn=io.input_fn, output=io.output)
        start_mock.assert_called_once()
        self.assertEqual(send_mock.call_count, 2)
        for call in send_mock.call_args_list:
            self.assertIs(call.args[0], fake_session)


class FormatCountsChartTests(unittest.TestCase):
    def test_orders_by_frequency_and_shows_percentage(self):
        chart = chat_cli._format_counts_chart({"111": 512, "000": 512})
        lines = chart.splitlines()
        self.assertIn("1024", lines[0])
        self.assertIn("000", lines[1])
        self.assertIn("50.0%", lines[1])
        self.assertIn("111", lines[2])

    def test_smallest_nonzero_count_still_gets_a_visible_bar(self):
        chart = chat_cli._format_counts_chart({"0": 1023, "1": 1})
        self.assertIn("█", chart.splitlines()[-1])  # rounds to zero-width without the floor


REPLY_WITH_BELL_QASM = (
    "[TASK: generate]\n[TARGET: ghz]\n```\nOPENQASM 2.0;\n"
    'include "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n```\n"
)


class VisualizeCircuitResultTests(unittest.TestCase):
    def test_no_qasm_in_reply_returns_none(self):
        self.assertIsNone(chat_cli._visualize_circuit_result("[TASK: select_backend]\n用 xxx 吧"))

    def test_execution_failure_returns_none_instead_of_raising(self):
        with mock.patch.object(
            chat_cli.l2_agent, "_execute_via_spinqit", side_effect=RuntimeError("boom")
        ):
            self.assertIsNone(chat_cli._visualize_circuit_result(REPLY_WITH_BELL_QASM))

    def test_real_execution_produces_a_chart(self):
        # Uses the real spinqit execution path (no mocking) to prove the
        # whole chain -- extract_qasm -> _execute_via_spinqit -> chart
        # formatting -- actually works together, not just each piece in
        # isolation.
        chart = chat_cli._visualize_circuit_result(REPLY_WITH_BELL_QASM)
        self.assertIsNotNone(chart)
        self.assertIn(f"{chat_cli._VISUALIZATION_SHOTS} 次", chart)
        # Bell state: only 00/11 should ever appear as a bitstring line,
        # never 01/10 -- checked on the raw counts, not fragile substring
        # matching against the formatted chart (e.g. "1024" contains "10").
        qasm = chat_cli.l2_agent.extract_qasm(REPLY_WITH_BELL_QASM)
        raw_counts = chat_cli.l2_agent._execute_via_spinqit(qasm, chat_cli._VISUALIZATION_SHOTS)
        self.assertEqual(set(raw_counts.keys()), {"00", "11"})


if __name__ == "__main__":
    unittest.main()
