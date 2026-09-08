from __future__ import annotations

from waf_proxy.config import get_settings
from waf_proxy.decision import create_app
from waf_proxy.events import EventStore

settings = get_settings()
app = create_app(settings, EventStore(settings.events_db))
