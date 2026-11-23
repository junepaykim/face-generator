; facegen's Windows installer (Phase 13), for Inno Setup 6. The release workflow builds it:
;
;   iscc /DVersion=0.1.0 /DStage=build\stage /DOutput=build\dist installer\facegen.iss
;
; Stage is installer/stage.py's folder: the release payload (the same files an update in place
; writes), .bin\uv.exe and facegen.ico.
;
; Per user, no administrator: the app goes to %LOCALAPPDATA%\facegen, as its data folder must be
; writable (not Program Files). Python, PyTorch and the models are not in it: they would make a
; 15 GB installer, so Start.bat fetches them at the first start (asking before the models).
; The shortcuts start Start.bat; once the first setup has made .venv, `facegen launch` points
; them at the app itself (desktop/shortcuts.py, refresh), so later starts show no console.
; Uninstalling leaves FM27's graphics\facegen\ and facegen's data folder unless the user says
; otherwise, and a silent uninstall keeps both.
;
; Saved as UTF-8 with a byte-order mark, so Inno Setup reads the Korean texts as UTF-8.

#ifndef Version
  #error Pass the version: iscc /DVersion=<major.minor.patch> installer\facegen.iss
#endif
#ifndef Stage
  #define Stage "..\build\stage"
#endif
#ifndef Output
  #define Output "..\build\dist"
#endif
#define KoreanIsl AddBackslash(CompilerPath) + "Languages\Korean.isl"

[Setup]
; Never change AppId: an install of a newer version finds and replaces the older one by it.
AppId={{328123DD-8BAD-4773-B1A0-9BC0E40350F2}
AppName=facegen
AppVersion={#Version}
AppVerName=facegen {#Version}
AppPublisher=facegen
AppPublisherURL=https://github.com/junepaykim/face-generator
AppSupportURL=https://github.com/junepaykim/face-generator/issues
AppUpdatesURL=https://github.com/junepaykim/face-generator/releases
VersionInfoVersion={#Version}
DefaultDirName={localappdata}\facegen
DisableDirPage=auto
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#Output}
OutputBaseFilename=facegen-setup-{#Version}
SetupIconFile={#Stage}\facegen.ico
UninstallDisplayIcon={app}\facegen.ico
UninstallDisplayName=facegen
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The app holds this mutex while it runs (desktop/shell.py): Setup and the uninstaller ask the
; user to quit facegen first rather than replacing files in use.
AppMutex=facegen-app
CloseApplications=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
#if FileExists(KoreanIsl)
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"
#endif

[CustomMessages]
en.DesktopIcon=Create a desktop shortcut
en.StartNow=Start facegen now (the first start sets up Python and its libraries, about 3 GB)
en.UpdateShortcut=Update facegen
en.RemovePack=Also remove facegen's pictures from FM27?%n%nThis deletes the graphics\facegen folder in FM27's folder, and FM27 shows its own photos again. Nothing else in FM27's folder is touched.
en.RemovePackFailed=facegen's pictures could not be removed from FM27. You can delete the graphics\facegen folder in FM27's folder yourself.
en.RemoveData=Also delete facegen's data?%n%n%1 holds the pictures, the database and your settings. Choose No to keep them for a later install.
#if FileExists(KoreanIsl)
ko.DesktopIcon=바탕 화면에 바로 가기 만들기
ko.StartNow=지금 facegen 시작(처음 시작할 때 Python과 라이브러리 약 3GB를 설치합니다)
ko.UpdateShortcut=facegen 업데이트
ko.RemovePack=FM27에서 facegen 그림도 지울까요?%n%nFM27 폴더의 graphics\facegen 폴더를 지우고, FM27은 원래 사진을 다시 보여 줍니다. FM27 폴더의 다른 파일은 건드리지 않습니다.
ko.RemovePackFailed=FM27에서 facegen 그림을 지우지 못했습니다. FM27 폴더의 graphics\facegen 폴더를 직접 지우셔도 됩니다.
ko.RemoveData=facegen 데이터도 지울까요?%n%n%1 폴더에 그림, 데이터베이스와 설정이 있습니다. 나중에 다시 설치할 때 쓰려면 '아니요'를 누르세요.
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"

[InstallDelete]
; Installing over an older version: its source goes first, so no module it no longer has stays.
Type: filesandordirs; Name: "{app}\src"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{userprograms}\facegen"; Filename: "{app}\Start.bat"; WorkingDir: "{app}"; IconFilename: "{app}\facegen.ico"; Comment: "facegen: anime or photo portraits for FM27's players"
Name: "{userprograms}\{cm:UpdateShortcut}"; Filename: "{app}\Start.bat"; Parameters: "--update"; WorkingDir: "{app}"; IconFilename: "{app}\facegen.ico"
Name: "{userdesktop}\facegen"; Filename: "{app}\Start.bat"; WorkingDir: "{app}"; IconFilename: "{app}\facegen.ico"; Comment: "facegen: anime or photo portraits for FM27's players"; Tasks: desktopicon

[Run]
Filename: "{app}\Start.bat"; Description: "{cm:StartNow}"; WorkingDir: "{app}"; Flags: postinstall nowait shellexec skipifsilent

[UninstallDelete]
; What the first start made beside the installed files: Python's environment and caches.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\.bin"
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\native"
; Shortcuts the app wrote itself: Start with Windows, and Start.bat --shortcut's.
Type: files; Name: "{userstartup}\facegen.lnk"
Type: files; Name: "{userdesktop}\facegen.lnk"
Type: files; Name: "{userprograms}\facegen.lnk"

[Code]
var
  RemoveData: Boolean;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Python: String;
  Code: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    RemoveData := False;
    { A silent uninstall keeps both. }
    if UninstallSilent then
      Exit;
    { facegen itself finds FM27's folder and removes only graphics\facegen (facegen remove-pack). }
    Python := ExpandConstant('{app}\.venv\Scripts\python.exe');
    if FileExists(Python) and
       (MsgBox(CustomMessage('RemovePack'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
    begin
      if not Exec(Python, '-m facegen.cli remove-pack --yes', ExpandConstant('{app}'), SW_HIDE,
                  ewWaitUntilTerminated, Code) or (Code <> 0) then
        MsgBox(CustomMessage('RemovePackFailed'), mbError, MB_OK);
    end;
    if DirExists(ExpandConstant('{app}\data')) then
      RemoveData := MsgBox(FmtMessage(CustomMessage('RemoveData'), [ExpandConstant('{app}\data')]),
                           mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  end;
  if (CurUninstallStep = usPostUninstall) and RemoveData then
  begin
    DelTree(ExpandConstant('{app}\data'), True, True, True);
    DeleteFile(ExpandConstant('{app}\config\local.toml'));
    RemoveDir(ExpandConstant('{app}\config'));
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
