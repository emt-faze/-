from __future__ import annotations


def list_window_titles() -> list[str]:
    try:
        import win32gui
    except ImportError:
        return []
    titles: list[str] = []

    def callback(handle: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(handle):
            return True
        title = win32gui.GetWindowText(handle).strip()
        if title and title not in titles:
            titles.append(title)
        return True

    win32gui.EnumWindows(callback, None)
    return sorted(titles)
