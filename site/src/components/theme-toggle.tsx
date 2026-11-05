"use client";
import { useSyncExternalStore } from "react";
function subscribe(callback: () => void) {
  window.addEventListener("counterprior-theme", callback);
  return () => window.removeEventListener("counterprior-theme", callback);
}
function snapshot() {
  return document.documentElement.dataset.theme || "light";
}
export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, snapshot, () => "light");
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("counterprior-theme", next);
    } catch {}
    window.dispatchEvent(new Event("counterprior-theme"));
  }
  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        aria-hidden="true"
      >
        {theme === "dark" ? (
          <path d="M20.5 13.2A8.5 8.5 0 0 1 10.8 3.5 8.5 8.5 0 1 0 20.5 13.2Z" />
        ) : (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
          </>
        )}
      </svg>
      <span className="sr-only">{theme} theme</span>
    </button>
  );
}
