"""Translation validation and localhost routes, with fake clients and no hosted calls.

Run: python tests/test_translation.py
"""

from __future__ import annotations

from contextlib import contextmanager
import http.client
import http.server
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.service.translation import (  # noqa: E402
    MAX_REQUEST_BYTES, ProseTranslator, TranslationError, validate_request,
    validate_translation,
)


SOURCE = {
    "language": "hi",
    "summary": "Walking fell to 820 s from 1,810 s today (-54.7%). No fall was observed.",
    "recommendation": "Call today; arrange a review if walking has not recovered tomorrow.",
}
TRANSLATED = {
    "language": "hi",
    "summary": "आज चलने का समय 1,810 s से घटकर 820 s हो गया (-54.7%)। कोई गिरना नहीं देखा गया।",
    "recommendation": "आज फोन करें; यदि कल तक चलने में सुधार न हो, तो समीक्षा की व्यवस्था करें।",
}


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = json.dumps(TRANSLATED, ensure_ascii=False) if response is None else response
        self.error, self.calls = error, []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error:
            raise self.error
        return self.response


def load_web(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "web" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def server_for(handler):
    class QuietHandler(handler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def post(port, payload, headers=None, path="/translate"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", path, body=body,
                           headers=headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class TranslationValidationTests(unittest.TestCase):
    def test_valid_translation_preserves_source_and_only_requests_prose(self):
        original = dict(SOURCE)
        llm = FakeLLM()
        result = ProseTranslator(llm).translate(SOURCE)
        self.assertEqual(result, TRANSLATED)
        self.assertEqual(SOURCE, original)
        prompt, options = llm.calls[0]
        self.assertIn("negation", prompt)
        self.assertIn("urgency", prompt)
        self.assertIn("temporal qualifier", prompt)
        self.assertIn("not a new assessment", prompt)
        self.assertTrue(options["constrained"])
        self.assertEqual(set(options["schema"]["properties"]), set(SOURCE))

    def test_changed_removed_or_added_numbers_are_rejected_per_field(self):
        cases = [
            {**TRANSLATED, "summary": TRANSLATED["summary"].replace("820", "821")},
            {**TRANSLATED, "summary": TRANSLATED["summary"].replace("-54.7%", "54.7%")},
            {**TRANSLATED, "summary": TRANSLATED["summary"].replace("1,810", "1810")},
            {**TRANSLATED, "summary": TRANSLATED["summary"].replace("820", "८२०")},
            {**TRANSLATED, "recommendation": TRANSLATED["recommendation"] + " 820"},
            # The total numbers are unchanged, but the summary's figure moved to advice.
            {**TRANSLATED, "summary": TRANSLATED["summary"].replace("820", ""),
             "recommendation": TRANSLATED["recommendation"] + " 820"},
        ]
        for item in cases:
            with self.subTest(item=item), self.assertRaises(TranslationError) as exc:
                validate_translation(json.dumps(item, ensure_ascii=False), SOURCE)
            self.assertEqual(exc.exception.code, "translation_invalid")

    def test_malformed_incomplete_or_untranslated_output_fails(self):
        cases = [
            '{"language":"hi","summary":"अधूरा',
            "```json\n" + json.dumps(TRANSLATED) + "\n```",
            json.dumps({"language": "hi", "summary": "अधूरा"}),
            json.dumps({**TRANSLATED, "recommendation": ""}),
            json.dumps({**TRANSLATED, "language": "mr"}),
            json.dumps({**TRANSLATED, "claims": []}),
            json.dumps(SOURCE),
            '{"language":"hi","language":"hi","summary":"अ","recommendation":"ब"}',
            '{"language":"hi","summary":NaN,"recommendation":"ब"}',
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(TranslationError):
                validate_translation(raw, SOURCE)

    def test_request_rejects_bad_language_types_extra_claims_and_size(self):
        for item in [[], {**SOURCE, "language": "en"}, {**SOURCE, "language": []},
                     {**SOURCE, "summary": None}, {**SOURCE, "summary": "a" * 2001},
                     {**SOURCE, "summary": "a\x00b"}, {**SOURCE, "claims": []}]:
            with self.subTest(item=item), self.assertRaises(TranslationError):
                validate_request(item)

    def test_empty_fields_stay_empty_and_empty_report_skips_network(self):
        llm = FakeLLM()
        empty = {"language": "mr", "summary": "", "recommendation": ""}
        self.assertEqual(ProseTranslator(llm).translate(empty), empty)
        self.assertEqual(llm.calls, [])
        with self.assertRaises(TranslationError):
            validate_translation(json.dumps({**empty, "summary": "नवीन सल्ला"}), empty)

    def test_provider_failure_never_exposes_diagnostics_and_releases_lock(self):
        llm = FakeLLM(error=RuntimeError("private-provider-diagnostics"))
        translator = ProseTranslator(llm)
        with self.assertRaises(TranslationError) as exc:
            translator.translate(SOURCE)
        self.assertEqual(exc.exception.status, 503)
        self.assertNotIn("private-provider", str(exc.exception.payload()))
        llm.error = None
        self.assertEqual(translator.translate(SOURCE), TRANSLATED)

    def test_concurrent_translation_is_bounded_without_touching_client(self):
        llm = FakeLLM()
        translator = ProseTranslator(llm)
        translator._lock.acquire()
        try:
            with self.assertRaises(TranslationError) as exc:
                translator.translate(SOURCE)
            self.assertEqual(exc.exception.code, "translation_busy")
            self.assertEqual(llm.calls, [])
        finally:
            translator._lock.release()


class TranslationRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = load_web("local_backend")
        cls.dev = load_web("dev_backend")

    def test_local_route_works_without_upstream_or_reporter_calls(self):
        llm = FakeLLM()
        class Handler(self.local.Handler):
            translator = ProseTranslator(llm)
        with server_for(Handler) as port:
            status, payload = post(port, SOURCE)
        self.assertEqual(status, 200)
        self.assertEqual(payload, TRANSLATED)
        self.assertEqual(len(llm.calls), 1)

    def test_local_route_returns_safe_failure(self):
        class Handler(self.local.Handler):
            translator = ProseTranslator(FakeLLM(response="not json"))
        with server_for(Handler) as port:
            status, payload = post(port, SOURCE)
        self.assertEqual(status, 502)
        self.assertEqual(payload["error_code"], "translation_invalid")

    def test_both_routes_reject_bad_requests_before_translation(self):
        llm = FakeLLM()
        class Local(self.local.Handler):
            translator = ProseTranslator(llm)
        for handler in (Local, self.dev.Handler):
            with self.subTest(handler=handler), server_for(handler) as port:
                self.assertEqual(post(port, b"{bad")[0], 400)
                self.assertEqual(post(port, b"\xff")[0], 400)
                self.assertEqual(post(port, b'{"language":"hi","language":"mr","summary":"","recommendation":""}')[0], 400)
                self.assertEqual(post(port, {**SOURCE, "language": "en"})[0], 422)
                self.assertEqual(post(port, SOURCE, {"Content-Type": "text/plain"})[0], 415)
                self.assertEqual(post(port, b"x" * (MAX_REQUEST_BYTES + 1))[0], 413)
                self.assertEqual(post(port, b"{}", {"Content-Type": "application/json",
                                                    "Content-Length": "9" * 4500})[0], 413)
        self.assertEqual(llm.calls, [])

    def test_dev_translates_known_fixture_in_both_languages(self):
        original = dict(self.dev.DEMO)
        with server_for(self.dev.Handler) as port:
            for language in ("hi", "mr"):
                source = {"language": language, **{
                    field: self.dev.DEMO[field] for field in ("summary", "recommendation")}}
                status, payload = post(port, source)
                self.assertEqual(status, 200)
                self.assertEqual(payload["language"], language)
                self.assertNotEqual(payload["summary"], source["summary"])
        self.assertEqual(self.dev.DEMO, original)

    def test_rejections_deliver_json_when_body_is_sent_after_headers(self):
        # http.client normally sends a whole request together. Sending its body after
        # the headers reproduces the unread-kernel-bytes path that reset Windows
        # sockets and hid 415/413 JSON responses from the browser.
        class Local(self.local.Handler):
            translator = ProseTranslator(FakeLLM())
        for handler in (Local, self.dev.Handler):
            with self.subTest(handler=handler), server_for(handler) as port:
                for content_type, size, expected in (
                    ("text/plain", 500, 415),
                    ("application/json", MAX_REQUEST_BYTES + 1, 413),
                ):
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        connection.putrequest("POST", "/translate")
                        connection.putheader("Content-Type", content_type)
                        connection.putheader("Content-Length", str(size))
                        connection.endheaders()
                        connection.send(b"x" * size)
                        response = connection.getresponse()
                        self.assertEqual(response.status, expected)
                        self.assertEqual(json.loads(response.read())["error_code"], "invalid_request")
                    finally:
                        connection.close()

    def test_dev_does_not_invent_translation_for_unknown_report(self):
        with server_for(self.dev.Handler) as port:
            status, payload = post(port, SOURCE)
        self.assertEqual(status, 503)
        self.assertEqual(payload["error_code"], "fixture_translation_unavailable")
        self.assertIn("नमूना", payload["detail"])


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    completed = result.testsRun - len(result.skipped)
    failed = len(result.failures) + len(result.errors) + len(result.unexpectedSuccesses)
    print(f"{max(0, completed - failed)}/{completed} passed ({len(result.skipped)} skipped)")
    sys.exit(0 if result.wasSuccessful() else 1)
