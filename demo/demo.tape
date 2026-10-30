# The demo in the README. VHS types these keys into the real yardstick-predict and films the terminal.
# Run demo/record.py, not this file. It sets up the made-up history and the stand-in for Voyage first.

Output demo/demo.gif
Output demo/demo.mp4

Require yardstick-predict

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 24
Set Columns 80
Set Rows 30
Set Padding 24
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set BorderRadius 10
Set CursorBlink false
Set TypingSpeed 55ms
Set Framerate 24

# The command is typed before filming starts. So the first frame says what runs.
Hide
Type "yardstick-predict"
Show
Sleep 1.2s
Enter
Wait+Screen /What would you like to do/
Sleep 2.5s

# Estimate is the first choice.
Enter
Wait+Screen /Describe the task/
Sleep 1s
Type "Add a login page with tests"
Sleep 700ms
Enter

# The AI models with enough data, then tick both. VHS's Wait reads only the first screenful the terminal printed.
# From here, where the screen scrolls, pauses set the pace. A key pressed early still counts.
Wait+Screen /Which AI models should it forecast/
Sleep 1.5s
Enter
Sleep 1.2s
Space
Sleep 600ms
Down
Sleep 500ms
Space
Sleep 1s
Enter

Sleep 9s
