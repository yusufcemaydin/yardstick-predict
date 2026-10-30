# The command-line demo in the README. VHS types these commands into the real yardstick-predict and films them.
# Run demo/record.py, not this file. It sets up the made-up history and the stand-in for Voyage first.

Output demo/cli.gif
Output demo/cli.mp4

Require yardstick-predict
Require jq

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 20
Set Columns 80
Set Rows 46
Set Padding 24
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set BorderRadius 10
Set CursorBlink false
Set TypingSpeed 35ms
Set Framerate 24

# A person names the AI model and the thinking level. A table comes back.
Sleep 500ms
Type `yardstick-predict predict "Add a login page with tests" \`
Enter
Type "  --model claude-opus-5-5 --effort max"
Sleep 500ms
Enter
Sleep 5s

# A program adds --distribution and gets each forecast's curve in JSON. jq picks out the first forecast.
Ctrl+L
Sleep 500ms
Type `yardstick-predict predict "Add a login page with tests" \`
Enter
Type `  --model claude-opus-5-5 --effort max --distribution | jq '.forecasts[0]'`
Sleep 500ms
Enter
Sleep 10s
