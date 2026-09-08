import shutil
import subprocess
import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not installed")
def test_compose_config_is_valid():
    out = subprocess.run(["docker", "compose", "config"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "waf-proxy" in out.stdout
    # Compose v1 renders the short form; Compose v2 normalises to long form.
    assert "8080:8080" in out.stdout or 'published: "8080"' in out.stdout
