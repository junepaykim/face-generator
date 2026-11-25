"""The first frame of the Windows dashboard, while WebView2 loads the local page."""

from __future__ import annotations

import os
import threading
from typing import Any

_FIRST_PAINT = """
new Promise(resolve => {
    const observer = new PerformanceObserver(list => {
        if (list.getEntriesByName("first-contentful-paint").length) {
            observer.disconnect();
            resolve(true);
        }
    });
    observer.observe({type: "paint", buffered: true});
})
"""


def background(theme: str) -> str:
    """Match the dashboard's CSS, including Windows app mode for the Auto setting."""
    if theme == "auto" and os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                light, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                theme = "light" if light else "dark"
        except OSError:
            pass
    return "#13161b" if theme == "dark" else "#f5f6f8"


def cover_until_loaded(window: Any) -> None:
    """Called by pywebview's before_show on the WinForms thread, before its first paint."""
    from System import Action
    from System.Drawing import Bitmap, Point, Size
    from System.IO import MemoryStream
    from System.Windows.Forms import (
        AnchorStyles,
        Padding,
        PictureBox,
        PictureBoxSizeMode,
        ProgressBar,
        ProgressBarStyle,
        RowStyle,
        SizeType,
        TableLayoutPanel,
    )

    from kitshot.desktop.app_icon import bundled_png

    form = window.native
    scale = form.DeviceDpi / 96
    cover = TableLayoutPanel()
    # Keep the WebView visible around the mark so Chromium can paint underneath it.
    cover.Size = Size(round(120 * scale), round(100 * scale))
    cover.Location = Point(
        (form.ClientSize.Width - cover.Width) // 2, (form.ClientSize.Height - cover.Height) // 2
    )
    cover.Anchor = getattr(AnchorStyles, "None")
    cover.BackColor = form.BackColor
    cover.ColumnCount, cover.RowCount = 1, 2
    cover.RowStyles.Add(RowStyle(SizeType.Percent, 66))
    cover.RowStyles.Add(RowStyle(SizeType.Percent, 34))

    size = round(64 * scale)
    stream = MemoryStream(bundled_png(128))
    image = Bitmap(stream)
    logo = PictureBox()
    logo.Image = image
    logo.Size = Size(size, size)
    logo.SizeMode = PictureBoxSizeMode.Zoom
    logo.Anchor = AnchorStyles.Bottom
    logo.Margin = Padding(0)
    bar = ProgressBar()
    bar.Style = ProgressBarStyle.Marquee
    bar.Size = Size(round(96 * scale), round(6 * scale))
    bar.Anchor = AnchorStyles.Top
    bar.Margin = Padding(0, round(14 * scale), 0, 0)
    cover.Controls.Add(logo, 0, 0)
    cover.Controls.Add(bar, 0, 1)

    def remove() -> None:
        cover.Dispose()
        image.Dispose()
        stream.Dispose()

    def dismiss() -> None:
        window.events.loaded -= dismiss
        try:
            # WebView2 can finish navigation before its compositor presents any content.
            painted = threading.Event()
            window.evaluate_js(_FIRST_PAINT, callback=lambda _: painted.set())
            painted.wait(1)  # hidden windows and error documents may never report paint
        finally:
            if not form.IsDisposed and form.IsHandleCreated:
                form.BeginInvoke(Action(remove))

    window.events.loaded += dismiss
    form.Controls.Add(cover)
    cover.BringToFront()
