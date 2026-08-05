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


if __name__ == "__main__":
    unittest.main()
