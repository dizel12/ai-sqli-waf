from __future__ import annotations

import os

from waf_proxy.config import get_settings
from waf_proxy.decision import create_app
from waf_proxy.events import EventStore

settings = get_settings()
# Guarded like inference_svc.app: only build the app (and open the events DB) at
# import time when explicitly asked. The Dockerfile sets WAF_EAGER=1 for the
# uvicorn CMD; test code calls create_app() directly.
app = (create_app(settings, EventStore(settings.events_db))
       if os.environ.get("WAF_EAGER") else None)
