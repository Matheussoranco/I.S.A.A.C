import asyncio
from typing import Dict, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

class BrowserManager:
    """
    Lifecycle management for a local Chromium instance via Playwright.
    Handles session isolation by creating separate browser contexts for different profiles.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrowserManager, cls).__new__(cls)
            cls._instance._playwright = None
            cls._instance._browser = None
            cls._instance._contexts: Dict[str, BrowserContext] = {}
        return cls._instance

    async def start(self):
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        return self

    async def get_context(self, session_id: str = "default") -> BrowserContext:
        if not self._browser:
            await self.start()
        
        if session_id not in self._contexts:
            # Session isolation: separate contexts for separate profiles/sessions
            self._contexts[session_id] = await self._browser.new_context()
        
        return self._contexts[session_id]

    async def close_session(self, session_id: str = "default"):
        if session_id in self._contexts:
            await self._contexts[session_id].close()
            del self._contexts[session_id]

    async def shutdown(self):
        for context in self._contexts.values():
            await context.close()
        self._contexts.clear()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

# Global singleton for easy access across tools
browser_manager = BrowserManager()
