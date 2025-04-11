# projectMain


## For building/running the project on Windows

1. Download `vs_buildtools.exe` from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Navigate to Downloads/ folder (or wherever it downloaded to) through a terminal then run the command:
   `vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools`
3. After the build tools have finish installing, open a terminal in PyCharm navigated to the project and then run the command:
   `pip install flask torch torchaudio asteroid demucs ffmpeg-python soundfile`
4. After all of this you should be able to run the app.py file


The final amalgamation of each individual part of the project

(4/8/25 8:08 PM CST) there is an app.py that is a rough merge of Ethan's flask routes and the two audio separators in app.py, a copy of Jonny's datasetDownloader, all the uploads shared to the github, and a set of templates that are a nice starting place to begin finalizing the front end.

CURRENT (4/11/25 4:02 AM CST) Amy- Added GUI components, added installation instructions to
                                   the README.md, and did some code cleanup 