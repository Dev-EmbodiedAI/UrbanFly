#ifndef VERSION
  #define VERSION "1.0.0"
#endif
#ifndef SOURCEDIR
  #error SOURCEDIR must be supplied by build_release_windows.ps1
#endif
#ifndef OUTPUTDIR
  #error OUTPUTDIR must be supplied by build_release_windows.ps1
#endif

[Setup]
AppId={{6B5919BB-112D-4B34-9CF0-7533301A8B37}
AppName=UrbanFly
AppVersion={#VERSION}
AppPublisher=UrbanFly contributors
AppPublisherURL=https://github.com/Dev-EmbodiedAI/UrbanFly
AppSupportURL=https://github.com/Dev-EmbodiedAI/UrbanFly/issues
DefaultDirName={localappdata}\Programs\UrbanFly
DefaultGroupName=UrbanFly
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OUTPUTDIR}
OutputBaseFilename=UrbanFly-Windows-x64-{#VERSION}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\UrbanFly.exe
VersionInfoVersion={#VERSION}
VersionInfoProductName=UrbanFly Helsinki Digital Twin

[Languages]
Name: "chinesesimplified"; MessagesFile: ".\ChineseSimplified.isl"

[Files]
Source: "{#SOURCEDIR}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\UrbanFly"; Filename: "{app}\UrbanFly.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\UrbanFly"; Filename: "{app}\UrbanFly.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

[Run]
Filename: "{app}\UrbanFly.exe"; Description: "启动 UrbanFly 数字孪生"; Flags: nowait postinstall skipifsilent
