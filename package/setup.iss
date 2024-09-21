[Setup]
AppName=NanoGene
AppVersion=1.0
DefaultDirName={pf}\NanoGene
DefaultGroupName=NanoGene
OutputDir=.
OutputBaseFilename=NanoGene_installer
DisableDirPage=no
SetupIconFile=My.ico
[Files]
; Include the main executable
Source: "dist\NanoGene\NanoGene.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\NanoGene\config.toml"; DestDir: "{app}"; Flags: ignoreversion

; Include the entire folder and its contents
Source: "dist\NanoGene\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\NanoGene"; Filename: "{app}\NanoGene.exe";IconFilename: "{app}\icon\My.ico"
Name: "{group}\Uninstall NanoGene"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\NanoGene.exe"; Description: "{cm:LaunchProgram,NanoGene}"; Flags: nowait postinstall skipifsilent