[Setup]
AppName=NanoGA
AppVersion=1.0
DefaultDirName={pf}\NanoGA
DefaultGroupName=NanoGA
OutputDir=.
OutputBaseFilename=NanoGA_installer
DisableDirPage=no
SetupIconFile=My.ico
[Files]
; Include the main executable
Source: "dist\NanoGA\NanoGA.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\NanoGA\config.toml"; DestDir: "{app}"; Flags: ignoreversion

; Include the entire folder and its contents
Source: "dist\NanoGA\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\NanoGA"; Filename: "{app}\NanoGA.exe";IconFilename: "{app}\icon\My.ico"
Name: "{group}\Uninstall NanoGA"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\NanoGA.exe"; Description: "{cm:LaunchProgram,NanoGA}"; Flags: nowait postinstall skipifsilent