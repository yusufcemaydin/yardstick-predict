# The Claude Code demo. VHS starts Claude Code, types the task and films the real session.
# Run demo/record_claude_code.py, not this file. It sets up the made-up history, yardstick's hook and status line
# and the shop Claude works on. Afterwards it speeds up the middle.

Output raw.mp4

Require claude

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 20
Set Columns 80
Set Rows 30
Set Padding 24
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set BorderRadius 10
Set CursorBlink false
Set TypingSpeed 55ms
Set Framerate 12

# Claude Code starts out of sight. Filming starts once it's ready. Claude can't stop to ask a question or to have a
# plan approved, as no one is there to answer.
Hide
Type `clear && cd ~/storefront && claude --effort max --strict-mcp-config --no-chrome --disallowedTools AskUserQuestion EnterPlanMode`
Enter
Wait+Screen@60s /auto mode on/
Sleep 1s
Show

Sleep 1.5s
Type "Add a login page with tests"
Sleep 800ms
Enter

# Claude really works on it now. yardstick's line says how long it took once Claude stops.
Wait+Screen@25m /took \d+:\d\d/
Sleep 7s

# Claude Code closes the usual way. Killed in fullscreen mode, it would turn that mode off next time.
Hide
Ctrl+C
Sleep 500ms
Ctrl+C
Sleep 2s
