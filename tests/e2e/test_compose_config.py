import shutil
import subprocess
import pytest
import yaml

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not installed")
def test_compose_config_is_valid():
    out = subprocess.run(["docker", "compose", "config"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "waf-proxy" in out.stdout

    cfg = yaml.safe_load(out.stdout)

    published = set()
    for svc in cfg.get("services", {}).values():
        for p in svc.get("ports", []) or []:
            if isinstance(p, dict):
                pub = p.get("published")
                host_ip = p.get("host_ip")
            else:  # short "host:container" / "ip:host:container" form
                bits = str(p).split(":")
                if len(bits) == 3:
                    host_ip, pub, _ = bits
                elif len(bits) == 2:
                    host_ip, pub = None, bits[0]
                else:
                    host_ip, pub = None, bits[0]
            if pub is None:
                continue
            # a missing / 0.0.0.0 host_ip means "published to every interface"
            host_ip = None if host_ip in (None, "", "0.0.0.0") else host_ip
            published.add((str(pub), host_ip))

    # waf-proxy(8080) and log-ui(8081) are host-wide; the dev override exposes
    # vuln-app(8000) on loopback only. Nothing else may be published.
    assert published == {
        ("8080", None),
        ("8081", None),
        ("8000", "127.0.0.1"),
    }, published
