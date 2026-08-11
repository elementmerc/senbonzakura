# Shared look for every recording on this site. Sourced by each tape, so the palette, the
# chrome and the typing speed are defined once and cannot drift between pages.
#
# The colours are the docs palette from docs/.vitepress/theme/palette.css, so a recording sits
# on the page as part of it rather than as a screenshot of somebody else's terminal.
#
# EVERYTHING HERE IS AT TWICE THE SIZE IT IS PUBLISHED AT. Each tape renders at 2x and
# tools/record-docs.sh scales the result down with lanczos, which is what supersampling is: the
# downscale averages several rendered pixels into each published one, so glyph edges come out
# smooth instead of stair-stepped. Measured on the track-audit clip, 2026-08-11: the supersampled
# GIF is both visibly crisper AND smaller, 89 KB against 103 KB, because smooth edges compress
# better than hard ones. Doubling the numbers below without doubling a tape's Width and Height
# would simply produce a very large font in a small window.

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 52
Set Padding 48
Set Margin 112
Set MarginFill "#3d4d6b"
Set BorderRadius 24
Set WindowBar Colorful
Set WindowBarSize 88
Set TypingSpeed 45ms
Set CursorBlink false
Set Framerate 30
Set WaitTimeout 90s

Set Theme {"background":"#12151b","foreground":"#d7dde8","cursor":"#c77b5a","selection":"#3d4d6b","black":"#2b3038","red":"#c77b5a","green":"#8fb98f","yellow":"#d6b48a","blue":"#6d81a6","magenta":"#9d8bb0","cyan":"#7fa8ad","white":"#b9c4d6","brightBlack":"#4a5364","brightRed":"#e0a488","brightGreen":"#a8cea8","brightYellow":"#e8cfa8","brightBlue":"#9db0d0","brightMagenta":"#b7a6c8","brightCyan":"#9cc4c9","brightWhite":"#eef1f6"}
