using System.IO;
using System.Threading;
using System.Windows;

namespace UrbanFly.Desktop;

public partial class App : Application
{
    private Mutex? _singleInstance;
    private bool _ownsMutex;

    protected override void OnStartup(StartupEventArgs e)
    {
        _singleInstance = new Mutex(true, "Local\\UrbanFly.DigitalTwin.Desktop", out var created);
        _ownsMutex = created;
        if (!created)
        {
            // Do not create a second renderer or steal focus with a modal.
            Shutdown();
            return;
        }

        DispatcherUnhandledException += (_, args) =>
        {
            try
            {
                var logDirectory = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "UrbanFly",
                    "Logs");
                Directory.CreateDirectory(logDirectory);
                File.AppendAllText(
                    Path.Combine(logDirectory, "desktop-crash.log"),
                    $"{DateTimeOffset.Now:O} {args.Exception}\n");
            }
            catch
            {
                // The crash reporter must never replace the original failure.
            }
        };

        base.OnStartup(e);
        var window = new MainWindow();
        MainWindow = window;
        if (Environment.GetEnvironmentVariable("URBANFLY_START_MINIMIZED") == "1")
        {
            window.WindowState = WindowState.Minimized;
            window.ShowInTaskbar = false;
        }
        window.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        if (_ownsMutex) _singleInstance?.ReleaseMutex();
        _singleInstance?.Dispose();
        base.OnExit(e);
    }
}
