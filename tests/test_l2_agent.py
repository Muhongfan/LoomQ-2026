import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from starter_kit import l2_agent
from starter_kit.evaluator import extract_qasm as evaluator_extract_qasm

BELL_QASM_REPLY = """好的，这是制备贝尔态并测量的电路：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```
"""


class ScriptedAPIHandler(BaseHTTPRequestHandler):
    reply_content = "mock reply"
    captured_payload = None

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).captured_payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": type(self).reply_content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockLLMServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedAPIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.environ_patch = mock.patch.dict(
            os.environ,
            {
                "LOOMQ_LLM_BASE_URL": "http://127.0.0.1:%d" % self.server.server_port,
                "LOOMQ_LLM_API_KEY": "local-key",
                "LOOMQ_LLM_MODEL": "local-model",
                "LOOMQ_LLM_TIMEOUT_SECONDS": "5",
            },
            clear=True,
        )
        self.environ_patch.start()

    def tearDown(self):
        self.environ_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class AgentChatRequestShapeTests(MockLLMServerTestCase):
    def test_sends_system_and_user_messages(self):
        ScriptedAPIHandler.reply_content = "ok"
        l2_agent.agent_chat("生成一个 3 比特 GHZ 态")

        payload = ScriptedAPIHandler.captured_payload
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(
            payload["messages"][1], {"role": "user", "content": "生成一个 3 比特 GHZ 态"}
        )

    def test_system_prompt_states_the_gate_whitelist(self):
        ScriptedAPIHandler.reply_content = "ok"
        l2_agent.agent_chat("hello")

        system_content = ScriptedAPIHandler.captured_payload["messages"][0]["content"]
        for gate in ("h", "cx", "cu1", "ccx", "swap"):
            self.assertIn(gate, system_content)

    def test_real_backend_table_is_injected_not_a_placeholder(self):
        ScriptedAPIHandler.reply_content = "ok"
        l2_agent.agent_chat("hello")

        system_content = ScriptedAPIHandler.captured_payload["messages"][0]["content"]
        self.assertIn("spinq_taurus_simulator", system_content)
        self.assertIn("originq_wukong", system_content)

    def test_returns_the_model_reply_content(self):
        ScriptedAPIHandler.reply_content = "the actual reply"
        result = l2_agent.agent_chat("anything")
        self.assertEqual(result, "the actual reply")


class GenerationRoundTripTests(MockLLMServerTestCase):
    def test_mock_generation_reply_is_extractable_end_to_end(self):
        """Full pipeline check: mock LLM -> agent_chat -> text containing a
        fenced OpenQASM 2.0 block -> extractable by evaluator.py's own
        extract_qasm (the function the real grading harness / evaluate_l2
        uses), not just our own duplicate of it."""
        ScriptedAPIHandler.reply_content = BELL_QASM_REPLY
        reply = l2_agent.agent_chat("生成一个贝尔态")

        via_evaluator = evaluator_extract_qasm(reply)
        via_l2_agent = l2_agent.extract_qasm(reply)
        self.assertIsNotNone(via_evaluator)
        self.assertEqual(via_evaluator, via_l2_agent)
        self.assertTrue(via_evaluator.startswith("OPENQASM 2.0;"))

        from starter_kit.circuit_ir import parse_qasm2
        from starter_kit.validator import validate_circuit

        circuit = parse_qasm2(via_evaluator)
        validate_circuit(circuit)  # must not raise


class MissingEnvironmentTests(unittest.TestCase):
    def test_missing_environment_raises_and_does_not_fall_back_to_mock_data(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LOOMQ_LLM_BASE_URL"):
                l2_agent.agent_chat("hello")


class TimeBudgetTests(unittest.TestCase):
    def test_remaining_decreases_over_time(self):
        budget = l2_agent.TimeBudget(1.0)
        first = budget.remaining()
        time.sleep(0.05)
        second = budget.remaining()
        self.assertLess(second, first)

    def test_remaining_floors_at_zero(self):
        budget = l2_agent.TimeBudget(0.01)
        time.sleep(0.05)
        self.assertEqual(budget.remaining(), 0.0)

    def test_next_call_timeout_divides_remaining_time(self):
        budget = l2_agent.TimeBudget(30.0)
        no_reserve = budget.next_call_timeout(reserved_for_more_calls=0)
        with_reserve = budget.next_call_timeout(reserved_for_more_calls=2)
        self.assertGreater(no_reserve, with_reserve)
        self.assertAlmostEqual(with_reserve, no_reserve / 3, delta=0.5)

    def test_next_call_timeout_respects_minimum(self):
        budget = l2_agent.TimeBudget(1.0)
        timeout = budget.next_call_timeout(reserved_for_more_calls=10, min_timeout=5.0)
        self.assertEqual(timeout, 5.0)


class TemporaryTimeoutTests(unittest.TestCase):
    def test_restores_previous_value(self):
        with mock.patch.dict(os.environ, {"LOOMQ_LLM_TIMEOUT_SECONDS": "42"}):
            with l2_agent._temporary_timeout(7.5):
                self.assertEqual(os.environ["LOOMQ_LLM_TIMEOUT_SECONDS"], repr(7.5))
            self.assertEqual(os.environ["LOOMQ_LLM_TIMEOUT_SECONDS"], "42")

    def test_restores_unset_state(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with l2_agent._temporary_timeout(7.5):
                self.assertIn("LOOMQ_LLM_TIMEOUT_SECONDS", os.environ)
            self.assertNotIn("LOOMQ_LLM_TIMEOUT_SECONDS", os.environ)


class SequencedAPIHandler(BaseHTTPRequestHandler):
    """Returns a different scripted reply on each successive call, so the
    retry loop's second (corrected) attempt can be distinguished from its
    first (broken) one."""

    reply_sequence = []
    call_count = 0
    captured_payloads = []

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).captured_payloads.append(payload)
        index = min(type(self).call_count, len(type(self).reply_sequence) - 1)
        content = type(self).reply_sequence[index]
        type(self).call_count += 1
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SequencedMockLLMServerTestCase(unittest.TestCase):
    def setUp(self):
        SequencedAPIHandler.call_count = 0
        SequencedAPIHandler.captured_payloads = []
        SequencedAPIHandler.reply_sequence = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SequencedAPIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.environ_patch = mock.patch.dict(
            os.environ,
            {
                "LOOMQ_LLM_BASE_URL": "http://127.0.0.1:%d" % self.server.server_port,
                "LOOMQ_LLM_API_KEY": "local-key",
                "LOOMQ_LLM_MODEL": "local-model",
                "LOOMQ_LLM_TIMEOUT_SECONDS": "5",
            },
            clear=True,
        )
        self.environ_patch.start()

    def tearDown(self):
        self.environ_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


# h only, no entanglement -- fidelity against the GHZ-3 ideal {"000":0.5,"111":0.5}
# will be far below 0.97 (q1/q2 never become |1>), reliably triggering a retry.
BROKEN_GHZ3_REPLY = """这是电路：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
measure q -> c;
```
"""

FIXED_GHZ3_REPLY = """修好了：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
```
"""

GHZ3_PROMPT = "生成一个 3 比特 GHZ 态并进行全测量"


class RetryLoopTests(SequencedMockLLMServerTestCase):
    def test_retries_after_failed_verification_and_returns_fixed_circuit(self):
        SequencedAPIHandler.reply_sequence = [BROKEN_GHZ3_REPLY, FIXED_GHZ3_REPLY]
        reply = l2_agent.agent_chat(GHZ3_PROMPT)

        self.assertEqual(SequencedAPIHandler.call_count, 2)
        self.assertEqual(l2_agent.extract_qasm(reply), l2_agent.extract_qasm(FIXED_GHZ3_REPLY))

    def test_second_request_includes_corrective_feedback(self):
        SequencedAPIHandler.reply_sequence = [BROKEN_GHZ3_REPLY, FIXED_GHZ3_REPLY]
        l2_agent.agent_chat(GHZ3_PROMPT)

        second_request_messages = SequencedAPIHandler.captured_payloads[1]["messages"]
        # system, user(original), assistant(broken reply), user(corrective feedback)
        self.assertEqual(len(second_request_messages), 4)
        self.assertEqual(second_request_messages[2]["content"], BROKEN_GHZ3_REPLY)
        self.assertIn("没有通过校验", second_request_messages[3]["content"])

    def test_correct_circuit_on_first_try_does_not_retry(self):
        SequencedAPIHandler.reply_sequence = [FIXED_GHZ3_REPLY]
        l2_agent.agent_chat(GHZ3_PROMPT)
        self.assertEqual(SequencedAPIHandler.call_count, 1)

    def test_gives_up_after_max_attempts_and_returns_last_reply(self):
        SequencedAPIHandler.reply_sequence = [BROKEN_GHZ3_REPLY] * l2_agent.MAX_GENERATION_ATTEMPTS
        reply = l2_agent.agent_chat(GHZ3_PROMPT)

        self.assertEqual(SequencedAPIHandler.call_count, l2_agent.MAX_GENERATION_ATTEMPTS)
        self.assertEqual(reply, BROKEN_GHZ3_REPLY)

    def test_reply_without_qasm_is_returned_immediately_without_verification(self):
        SequencedAPIHandler.reply_sequence = ["建议使用 spinq_taurus_simulator，满足零排队要求。"]
        reply = l2_agent.agent_chat("我需要运行一个 15 比特电路，且零排队等待，选哪个平台？")

        self.assertEqual(SequencedAPIHandler.call_count, 1)
        self.assertIn("spinq_taurus_simulator", reply)


class ConstraintExtractionTests(unittest.TestCase):
    """The three official examples from backend_capabilities.md, used
    verbatim as acceptance tests for the deterministic constraint
    extraction + matching logic."""

    def setUp(self):
        self.backends = l2_agent._load_backend_capabilities()

    def test_fifteen_qubit_zero_queue_matches_official_answer_set(self):
        constraints = l2_agent._extract_backend_constraints(
            "我需要运行一个 15 比特电路，且零排队等待，选哪个平台？"
        )
        candidates = {b["id"] for b in l2_agent._matching_backends(constraints, self.backends)}
        self.assertEqual(
            candidates, {"spinq_taurus_simulator", "originq_local_simulator", "braket_local_simulator"}
        )

    def test_five_qubit_real_hardware_no_cost_matches_official_answer_set(self):
        constraints = l2_agent._extract_backend_constraints("在真实量子硬件上跑一个 5 比特电路，不想花钱")
        candidates = {b["id"] for b in l2_agent._matching_backends(constraints, self.backends)}
        self.assertEqual(candidates, {"spinq_cloud_qpu", "originq_wukong"})

    def test_exceeding_every_backend_yields_no_candidates(self):
        constraints = l2_agent._extract_backend_constraints("我需要一个 100 比特的电路，有什么建议？")
        candidates = l2_agent._matching_backends(constraints, self.backends)
        self.assertEqual(candidates, [])

    def test_english_qubit_phrasing_is_also_recognized(self):
        constraints = l2_agent._extract_backend_constraints(
            "I need to run a 15-qubit circuit with zero queue, which platform should I use?"
        )
        self.assertEqual(constraints["min_qubits"], 15)
        self.assertTrue(constraints["queue_none"])

    def test_ambiguous_prompt_extracts_no_constraints(self):
        self.assertEqual(l2_agent._extract_backend_constraints("你好，能帮帮我吗？"), {})


class BackendSelectionSafetyNetTests(unittest.TestCase):
    def test_correct_llm_answer_is_returned_unchanged(self):
        reply = "推荐使用 spinq_taurus_simulator，它没有排队且比特数足够。"
        result = l2_agent._apply_backend_selection_safety_net(
            "我需要运行一个 15 比特电路，且零排队等待，选哪个平台？", reply
        )
        self.assertEqual(result, reply)

    def test_wrong_llm_answer_gets_corrected_to_a_valid_id(self):
        reply = "推荐使用 braket_cloud，它性能很强。"  # paid, wrong for a zero-queue-free ask
        result = l2_agent._apply_backend_selection_safety_net(
            "我需要运行一个 15 比特电路，且零排队等待，选哪个平台？", reply
        )
        self.assertIn(
            l2_agent._BACKEND_ID_RE.search(result).group(1),
            {"spinq_taurus_simulator", "originq_local_simulator", "braket_local_simulator"},
        )

    def test_missing_id_in_llm_answer_gets_corrected(self):
        reply = "这个应该用本地模拟器就可以了，具体哪个都行。"
        result = l2_agent._apply_backend_selection_safety_net(
            "在真实量子硬件上跑一个 5 比特电路，不想花钱", reply
        )
        found = set(l2_agent._BACKEND_ID_RE.findall(result))
        self.assertTrue(found & {"spinq_cloud_qpu", "originq_wukong"})

    def test_no_backend_satisfies_gives_honest_answer_with_alternative(self):
        reply = "可以试试某个后端。"
        result = l2_agent._apply_backend_selection_safety_net(
            "我需要一个 100 比特的电路，有什么建议？", reply
        )
        self.assertIn("没有任何后端", result)
        self.assertIn("originq_wukong", result)  # largest max_qubits (72) in the table

    def test_ambiguous_prompt_trusts_llm_reply_unchanged(self):
        reply = "随便一个都行。"
        result = l2_agent._apply_backend_selection_safety_net("你好，能帮帮我吗？", reply)
        self.assertEqual(result, reply)


class BackendSelectionEndToEndTests(MockLLMServerTestCase):
    def test_agent_chat_corrects_a_wrong_backend_answer(self):
        ScriptedAPIHandler.reply_content = "用 braket_cloud 吧。"
        reply = l2_agent.agent_chat("我需要运行一个 15 比特电路，且零排队等待，选哪个平台？")
        found = set(l2_agent._BACKEND_ID_RE.findall(reply))
        self.assertTrue(
            found & {"spinq_taurus_simulator", "originq_local_simulator", "braket_local_simulator"}
        )


class TaskTagClassificationTests(unittest.TestCase):
    def test_recognizes_each_tag(self):
        self.assertEqual(l2_agent._classify_task_tag("[TASK: generate]\n..."), "generate")
        self.assertEqual(l2_agent._classify_task_tag("[TASK: fix]\n..."), "fix")
        self.assertEqual(l2_agent._classify_task_tag("[TASK: select_backend]\n..."), "select_backend")

    def test_case_and_leading_whitespace_insensitive(self):
        self.assertEqual(l2_agent._classify_task_tag("  [task: GENERATE]\nfoo"), "generate")

    def test_missing_tag_returns_none(self):
        self.assertIsNone(l2_agent._classify_task_tag("这是一段没有标记的回复"))

    def test_tag_not_on_first_line_is_not_recognized(self):
        # Must be the reply's own first line, not a mention buried later.
        self.assertIsNone(l2_agent._classify_task_tag("先说点别的。\n[TASK: generate]\n..."))


class ReplyContentExtractionTests(unittest.TestCase):
    def test_extracts_content_from_well_formed_response(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        self.assertEqual(l2_agent._extract_reply_content(response), "hello")

    def test_empty_choices_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "no choices"):
            l2_agent._extract_reply_content({"choices": []})

    def test_missing_message_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "no message content"):
            l2_agent._extract_reply_content({"choices": [{}]})

    def test_non_string_content_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "not a string"):
            l2_agent._extract_reply_content({"choices": [{"message": {"content": 42}}]})

    def test_non_dict_response_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "not a JSON object"):
            l2_agent._extract_reply_content(["unexpected", "list"])


# The two misclassification scenarios _classify_task_tag exists to fix:
# a backend-selection answer that happens to quote example QASM in its
# explanation, and a generation answer that forgot the code-block format.
BACKEND_SELECTION_WITH_EMBEDDED_CODE_SNIPPET = """[TASK: select_backend]
建议使用 spinq_taurus_simulator。举例来说，一个简单电路大概长这样：
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```
这个平台零排队、免费、比特数也够用。
"""

GENERATE_MISSING_CODE_BLOCK = "[TASK: generate]\n好的，我需要制备一个 3 比特 GHZ 态。"


class TaskTagRoutingTests(SequencedMockLLMServerTestCase):
    def test_backend_selection_reply_with_embedded_code_is_not_misread_as_generation(self):
        # Without the tag, extract_qasm would find the embedded snippet and
        # misroute this into circuit verification instead of the backend
        # safety net -- confirms the tag fix prevents that.
        SequencedAPIHandler.reply_sequence = [BACKEND_SELECTION_WITH_EMBEDDED_CODE_SNIPPET]
        reply = l2_agent.agent_chat("我需要运行一个 15 比特电路，且零排队等待，选哪个平台？")

        self.assertEqual(SequencedAPIHandler.call_count, 1)  # not retried as a broken circuit
        self.assertIn("spinq_taurus_simulator", reply)

    def test_generation_reply_missing_code_block_triggers_retry_not_silent_misroute(self):
        # Without the tag, "no QASM found" always meant "treat as backend
        # selection" -- silently returning this reply as if it answered a
        # backend question. With the tag, we know it was meant to generate
        # a circuit and failed the format requirement, so it must retry.
        SequencedAPIHandler.reply_sequence = [GENERATE_MISSING_CODE_BLOCK, FIXED_GHZ3_REPLY]
        reply = l2_agent.agent_chat(GHZ3_PROMPT)

        self.assertEqual(SequencedAPIHandler.call_count, 2)
        self.assertEqual(l2_agent.extract_qasm(reply), l2_agent.extract_qasm(FIXED_GHZ3_REPLY))

    def test_second_request_after_missing_code_block_asks_for_the_format(self):
        SequencedAPIHandler.reply_sequence = [GENERATE_MISSING_CODE_BLOCK, FIXED_GHZ3_REPLY]
        l2_agent.agent_chat(GHZ3_PROMPT)

        second_request_messages = SequencedAPIHandler.captured_payloads[1]["messages"]
        self.assertIn("没有找到符合格式要求", second_request_messages[3]["content"])


class TargetTagClassificationTests(unittest.TestCase):
    def test_recognizes_each_variant(self):
        self.assertEqual(l2_agent._classify_target_tag("[TARGET: ghz]\n..."), "ghz")
        self.assertEqual(l2_agent._classify_target_tag("[TARGET: bell]\n..."), "bell")
        self.assertEqual(l2_agent._classify_target_tag("[TARGET: uniform]\n..."), "uniform")
        self.assertEqual(l2_agent._classify_target_tag("[TARGET: basis:101]\n..."), "basis:101")
        self.assertEqual(l2_agent._classify_target_tag("[TARGET: other]\n..."), "other")

    def test_missing_tag_returns_none(self):
        self.assertIsNone(l2_agent._classify_target_tag("没有任何标记的回复"))


class TargetRecognitionTests(unittest.TestCase):
    def test_tag_is_trusted_even_when_prompt_has_no_matching_keywords(self):
        # Deliberately avoids "GHZ"/"贝尔"/"纠缠"/"叠加" so the prompt-based
        # fallback alone would find nothing -- proves the tag does real work.
        prompt = "帮我做一个三个粒子永远同增同减的量子态"
        reply = "[TASK: generate]\n[TARGET: ghz]\n..."
        ideal, family = l2_agent._recognize_target(prompt, reply, n_clbits=3)
        self.assertEqual(family, "ghz")
        self.assertEqual(ideal, {"000": 0.5, "111": 0.5})

    def test_basis_tag_with_wrong_length_falls_back_to_prompt(self):
        # basis:1 doesn't match a 3-clbit circuit -- must not be trusted as-is.
        reply = "[TASK: generate]\n[TARGET: basis:1]\n..."
        ideal, family = l2_agent._recognize_target("随便生成点什么", reply, n_clbits=3)
        self.assertIsNone(ideal)
        self.assertIsNone(family)

    def test_other_tag_falls_back_to_prompt_based_inference(self):
        reply = "[TASK: generate]\n[TARGET: other]\n..."
        ideal, family = l2_agent._recognize_target("生成一个 GHZ 态", reply, n_clbits=3)
        self.assertEqual(family, "ghz")

    def test_no_tag_uses_prompt_based_fallback(self):
        ideal, family = l2_agent._recognize_target("生成一个贝尔态", "没有标记的回复", n_clbits=2)
        self.assertEqual(family, "ghz")
        self.assertEqual(ideal, {"00": 0.5, "11": 0.5})


class StructuralSanityCheckTests(unittest.TestCase):
    def _circuit(self, qasm: str):
        from starter_kit.circuit_ir import parse_qasm2

        return parse_qasm2(qasm)

    def test_qubit_count_mismatch_is_caught(self):
        circuit = self._circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nmeasure q -> c;\n'
        )
        issue = l2_agent._structural_sanity_check("生成一个 3 比特电路", circuit)
        self.assertIn("3 个比特", issue)
        self.assertIn("2 个量子比特", issue)

    def test_matching_qubit_count_is_not_flagged(self):
        circuit = self._circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\nmeasure q -> c;\n'
        )
        self.assertIsNone(l2_agent._structural_sanity_check("生成一个 3 比特电路", circuit))

    def test_claimed_entanglement_with_no_entangling_gate_is_caught(self):
        circuit = self._circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nmeasure q -> c;\n'
        )
        issue = l2_agent._structural_sanity_check("我想要一个纠缠态", circuit)
        self.assertIn("纠缠", issue)

    def test_entanglement_with_a_two_qubit_gate_is_not_flagged(self):
        circuit = self._circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
        )
        self.assertIsNone(l2_agent._structural_sanity_check("我想要一个纠缠态", circuit))

    def test_unrelated_prompt_is_not_flagged(self):
        circuit = self._circuit(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q -> c;\n'
        )
        self.assertIsNone(l2_agent._structural_sanity_check("生成一个随便的电路", circuit))


class FidelityFeedbackTests(unittest.TestCase):
    def test_format_distribution_orders_by_probability_and_caps_length(self):
        dist = {"00": 0.05, "11": 0.5, "01": 0.45}
        formatted = l2_agent._format_distribution(dist)
        self.assertTrue(formatted.startswith("{11: 50.0%, 01: 45.0%, 00: 5.0%}"))

    def test_fidelity_failure_feedback_includes_observed_ideal_and_hint(self):
        observed = {"000": 0.5, "001": 0.5}
        ideal = {"000": 0.5, "111": 0.5}
        message = l2_agent._fidelity_failure_feedback("ghz", observed, ideal, 0.5)
        self.assertIn("0.500", message)
        self.assertIn("000: 50.0%", message)
        self.assertIn("111: 50.0%", message)
        self.assertIn("纠缠门", message)

    def test_unknown_family_omits_hint_gracefully(self):
        message = l2_agent._fidelity_failure_feedback(None, {"0": 1.0}, {"1": 1.0}, 0.0)
        self.assertIn("保真度不足", message)


BROKEN_GHZ3_TAGGED_REPLY = """[TASK: generate]
[TARGET: ghz]
这是电路：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
measure q -> c;
```
"""

FIXED_GHZ3_TAGGED_REPLY = """[TASK: generate]
[TARGET: ghz]
修好了：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
```
"""


class RicherFeedbackIntegrationTests(SequencedMockLLMServerTestCase):
    def test_retry_feedback_includes_distributions_and_family_hint(self):
        SequencedAPIHandler.reply_sequence = [BROKEN_GHZ3_TAGGED_REPLY, FIXED_GHZ3_TAGGED_REPLY]
        l2_agent.agent_chat(GHZ3_PROMPT)

        second_request_messages = SequencedAPIHandler.captured_payloads[1]["messages"]
        feedback = second_request_messages[3]["content"]
        self.assertIn("实际测量分布", feedback)
        self.assertIn("目标分布应为", feedback)
        self.assertIn("纠缠门", feedback)

    def test_structural_mismatch_triggers_retry_even_without_recognized_family(self):
        # "3 比特" stated but the circuit only declares 2 -- no GHZ/Bell/etc
        # keyword present, so this only gets caught by the structural check.
        wrong_count_reply = """[TASK: generate]
[TARGET: other]
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
measure q -> c;
```
"""
        fixed_reply = """[TASK: generate]
[TARGET: other]
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
h q[2];
measure q -> c;
```
"""
        SequencedAPIHandler.reply_sequence = [wrong_count_reply, fixed_reply]
        reply = l2_agent.agent_chat("生成一个 3 比特电路，每个比特都做叠加")

        self.assertEqual(SequencedAPIHandler.call_count, 2)
        second_request_messages = SequencedAPIHandler.captured_payloads[1]["messages"]
        self.assertIn("3 个比特", second_request_messages[3]["content"])
        self.assertIn("2 个量子比特", second_request_messages[3]["content"])


if __name__ == "__main__":
    unittest.main()
