import os
import pytest

pytestmark = pytest.mark.e2e

if os.environ.get("RUN_E2E") != "1":
    pytest.skip("set RUN_E2E=1 to run", allow_module_level=True)

from scripts.demo import run_scenario, false_positive_count


def test_scenario1_vulnerable_no_waf_leaks():
    r = run_scenario(app_secure=False, waf_on=False)
    assert r["search_leaked_users"] is True
    assert r["login_status"] == 200


def test_scenario2_vulnerable_with_waf_blocks():
    r = run_scenario(app_secure=False, waf_on=True)
    assert r["search_leaked_users"] is False
    assert r["search_status"] == 403


def test_scenario3_secure_no_waf_blocks():
    r = run_scenario(app_secure=True, waf_on=False)
    assert r["search_leaked_users"] is False


def test_scenario4_secure_with_waf_blocks():
    r = run_scenario(app_secure=True, waf_on=True)
    assert r["search_leaked_users"] is False


def test_no_false_positives_on_benign_traffic():
    assert false_positive_count(n=60) == 0
