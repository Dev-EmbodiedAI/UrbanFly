using System.ComponentModel;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using Microsoft.Web.WebView2.Core;

namespace UrbanFly.Desktop;

public partial class MainWindow : Window
{
    private readonly BackendSupervisor _backend = new();
    private readonly CancellationTokenSource _lifetime = new();
    private readonly DispatcherTimer _healthTimer;
    private int _healthCheckActive;
    private int _consecutiveHealthFailures;
    private bool _closing;
    private bool _closeApproved;
    private bool _recoveringWebView;

    public MainWindow()
    {
        InitializeComponent();
        _healthTimer = new DispatcherTimer(TimeSpan.FromSeconds(2), DispatcherPriority.Background,
            async (_, _) => await CheckRuntimeHealthAsync(), Dispatcher);
        Loaded += async (_, _) => await StartAsync();
        StateChanged += (_, _) => PublishHostVisibility();
    }

    private async Task StartAsync()
    {
        var progress = new Progress<string>(message => StartupStatus.Text = message);
        try
        {
            await _backend.EnsureReadyAsync(progress, _lifetime.Token);
            await InitializeWebViewAsync();
            _healthTimer.Start();
        }
        catch (OperationCanceledException) when (_closing)
        {
        }
        catch (Exception error)
        {
            StartupStatus.Text = $"启动失败：{error.Message}";
        }
    }

    private async Task InitializeWebViewAsync()
    {
        StartupStatus.Text = "正在初始化 GPU 数字孪生界面…";
        var userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "UrbanFly",
            "WebView2");
        Directory.CreateDirectory(userData);
        var environment = await CoreWebView2Environment.CreateAsync(
            browserExecutableFolder: null,
            userDataFolder: userData);
        await TwinView.EnsureCoreWebView2Async(environment);
        var core = TwinView.CoreWebView2;
        core.Settings.AreDefaultContextMenusEnabled = false;
        core.Settings.AreDevToolsEnabled =
            Environment.GetEnvironmentVariable("URBANFLY_DESKTOP_DEVTOOLS") == "1";
        core.Settings.IsStatusBarEnabled = false;
        core.Settings.IsZoomControlEnabled = false;
        core.Settings.AreBrowserAcceleratorKeysEnabled = false;
        core.NavigationStarting += OnNavigationStarting;
        core.NavigationCompleted += OnNavigationCompleted;
        core.NewWindowRequested += (_, args) => args.Handled = true;
        core.PermissionRequested += (_, args) => args.State = CoreWebView2PermissionState.Deny;
        core.DownloadStarting += (_, args) => args.Cancel = true;
        core.ProcessFailed += OnWebViewProcessFailed;
        core.Navigate(new Uri(_backend.BaseUri, "?surface=desktop").AbsoluteUri);
    }

    private void OnNavigationStarting(object? sender, CoreWebView2NavigationStartingEventArgs args)
    {
        if (!Uri.TryCreate(args.Uri, UriKind.Absolute, out var target)
            || target.Scheme != _backend.BaseUri.Scheme
            || target.Host != "127.0.0.1"
            || target.Port != _backend.BaseUri.Port)
        {
            args.Cancel = true;
        }
    }

    private void OnNavigationCompleted(object? sender, CoreWebView2NavigationCompletedEventArgs args)
    {
        if (args.IsSuccess)
        {
            PublishHostVisibility();
            StartupOverlay.Visibility = Visibility.Collapsed;
            return;
        }
        StartupOverlay.Visibility = Visibility.Visible;
        StartupStatus.Text = $"界面加载失败：{args.WebErrorStatus}";
    }

    private void PublishHostVisibility()
    {
        if (_closing || TwinView.CoreWebView2 is null) return;
        var hidden = WindowState == WindowState.Minimized;
        TwinView.CoreWebView2.PostWebMessageAsJson(
            hidden ? "{\"type\":\"host_visibility\",\"hidden\":true}"
                : "{\"type\":\"host_visibility\",\"hidden\":false}");
    }

    private async void OnWebViewProcessFailed(object? sender, CoreWebView2ProcessFailedEventArgs args)
    {
        if (_closing || _recoveringWebView)
        {
            return;
        }
        _recoveringWebView = true;
        try
        {
            StartupOverlay.Visibility = Visibility.Visible;
            StartupStatus.Text = $"渲染进程异常，正在恢复… ({args.ProcessFailedKind})";
            await Task.Delay(500, _lifetime.Token);
            if (args.ProcessFailedKind == CoreWebView2ProcessFailedKind.BrowserProcessExited)
            {
                RootGrid.Children.Remove(TwinView);
                TwinView.Dispose();
                TwinView = new Microsoft.Web.WebView2.Wpf.WebView2();
                RootGrid.Children.Insert(0, TwinView);
                await InitializeWebViewAsync();
            }
            else if (args.ProcessFailedKind == CoreWebView2ProcessFailedKind.RenderProcessExited)
            {
                TwinView.CoreWebView2?.Reload();
            }
            else
            {
                StartupStatus.Text = "渲染暂时无响应；保留引擎和采集状态，等待恢复。";
            }
        }
        catch (OperationCanceledException) when (_closing) { }
        catch (Exception error) { StartupStatus.Text = $"渲染恢复失败：{error.Message}"; }
        finally { _recoveringWebView = false; }
    }

    private async Task CheckRuntimeHealthAsync()
    {
        if (Interlocked.Exchange(ref _healthCheckActive, 1) != 0)
        {
            return;
        }
        try
        {
            var health = await _backend.GetHealthAsync(_lifetime.Token);
            if (health.Healthy)
            {
                _consecutiveHealthFailures = 0;
                return;
            }

            _consecutiveHealthFailures += 1;
            if (_consecutiveHealthFailures < 3 || _closing)
            {
                return;
            }
            StartupOverlay.Visibility = Visibility.Visible;
            StartupStatus.Text = "实时引擎离线，正在自动恢复…";
            var progress = new Progress<string>(message => StartupStatus.Text = message);
            if (_backend.OwnsBackend)
            {
                StartupStatus.Text = "引擎响应超时但进程仍在运行；保留现场，不强制重启。";
                return;
            }
            else
            {
                await _backend.EnsureReadyAsync(progress, _lifetime.Token);
            }
            TwinView.CoreWebView2?.Reload();
            _consecutiveHealthFailures = 0;
        }
        catch (OperationCanceledException) when (_closing)
        {
        }
        catch (Exception error)
        {
            StartupStatus.Text = $"自动恢复失败：{error.Message}";
        }
        finally
        {
            Interlocked.Exchange(ref _healthCheckActive, 0);
        }
    }

    protected override async void OnClosing(CancelEventArgs e)
    {
        if (_closeApproved)
        {
            base.OnClosing(e);
            return;
        }
        e.Cancel = true;
        if (_closing) return;
        _closing = true;
        try
        {
            var health = await _backend.GetHealthAsync();
            var noSensorView = TwinView.CoreWebView2 is null;
            var engineAlreadyGone = !_backend.OwnsBackend && health.SimulatorState == "offline";
            if (!noSensorView && !engineAlreadyGone && (!health.Healthy || health.PolicyClients != 0))
            {
                StartupOverlay.Visibility = Visibility.Visible;
                StartupStatus.Text = health.Healthy
                    ? "采集器仍连接：请先停止采集，再关闭传感器界面。"
                    : "引擎健康状态未知：暂不关闭传感器界面，以保护采集现场。";
                _closing = false;
                return;
            }
        }
        catch
        {
            _closing = false;
            return;
        }
        _healthTimer.Stop();
        _lifetime.Cancel();
        try
        {
            await _backend.DisposeAsync();
        }
        catch
        {
            // The desktop shell must close even when a child process already exited.
        }
        TwinView.Dispose();
        _lifetime.Dispose();
        _closeApproved = true;
        _ = Dispatcher.BeginInvoke(Close);
    }
}
