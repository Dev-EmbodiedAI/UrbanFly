using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;

namespace UrbanFly.Desktop;

internal sealed record RuntimeHealth(
    bool Healthy,
    string SimulatorState,
    int? PolicyClients,
    int? TotalClients);

internal sealed class BackendSupervisor : IAsyncDisposable
{
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly SemaphoreSlim _startGate = new(1, 1);
    private readonly object _logGate = new();
    private Process? _ownedProcess;
    private StreamWriter? _logWriter;

    public BackendSupervisor(DirectoryInfo? projectRoot = null, Uri? baseUri = null)
    {
        ProjectRoot = projectRoot ?? LocateProjectRoot();
        BaseUri = baseUri ?? new Uri("http://127.0.0.1:8765/");
    }

    public DirectoryInfo ProjectRoot { get; }
    public Uri BaseUri { get; }
    public bool OwnsBackend => _ownedProcess is { HasExited: false };

    public async Task<bool> EnsureReadyAsync(
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _startGate.WaitAsync(cancellationToken);
        try
        {
            if ((await GetHealthAsync(cancellationToken)).Healthy)
            {
                progress?.Report("已连接 UrbanFly 实时引擎");
                return false;
            }

            if (_ownedProcess is { HasExited: false })
            {
                progress?.Report("引擎正在初始化 Helsinki 物理场景…");
            }
            else
            {
                using var probe = new TcpClient();
                try
                {
                    await probe.ConnectAsync(BaseUri.Host, BaseUri.Port, cancellationToken);
                    throw new InvalidOperationException(
                        "端口已被占用但健康状态未知；保留现有进程，不启动第二个后端。");
                }
                catch (SocketException)
                {
                    // No listening process: safe to start a new owned backend.
                }
                StartBackend();
                progress?.Report("正在启动 Helsinki 实时引擎…");
            }

            var deadline = DateTimeOffset.UtcNow.AddSeconds(45);
            while (DateTimeOffset.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if ((await GetHealthAsync(cancellationToken)).Healthy)
                {
                    progress?.Report("引擎就绪，正在载入数字孪生…");
                    return true;
                }
                if (_ownedProcess is { HasExited: true })
                {
                    throw new InvalidOperationException(
                        $"UrbanFly 后端启动失败，退出码 {_ownedProcess.ExitCode}。请查看 {LogPath()}");
                }
                await Task.Delay(250, cancellationToken);
            }
            throw new TimeoutException("UrbanFly 后端在 45 秒内未通过健康门禁。");
        }
        finally
        {
            _startGate.Release();
        }
    }

    public async Task<RuntimeHealth> GetHealthAsync(
        CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await _http.GetAsync(
                new Uri(BaseUri, "api/health"), cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                return new RuntimeHealth(false, "unavailable", null, null);
            }
            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var document = await JsonDocument.ParseAsync(
                stream,
                cancellationToken: cancellationToken);
            var root = document.RootElement;
            if (root.GetProperty("schema").GetString() != "urbanfly-runtime-health-v1")
            {
                return new RuntimeHealth(false, "incompatible-health-schema", null, null);
            }
            var status = root.GetProperty("status").GetString();
            var simulator = root.GetProperty("simulator");
            var clients = root.GetProperty("clients");
            return new RuntimeHealth(
                status == "ok",
                simulator.GetProperty("state").GetString() ?? "unknown",
                clients.GetProperty("policy").GetInt32(),
                clients.GetProperty("total").GetInt32());
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return new RuntimeHealth(false, "timeout", null, null);
        }
        catch (HttpRequestException)
        {
            return new RuntimeHealth(false, "offline", null, null);
        }
        catch (Exception error) when (error is JsonException or KeyNotFoundException
            or InvalidOperationException or FormatException)
        {
            return new RuntimeHealth(false, "invalid-health-payload", null, null);
        }
    }

    public async Task RestartOwnedBackendAsync(
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Health timeouts are NOT proof of death. Never kill a live process:
        // a collector may still be writing an episode while HTTP is degraded.
        if (OwnsBackend) throw new InvalidOperationException("引擎进程仍在运行；禁止超时强杀。");
        _ownedProcess?.Dispose();
        _ownedProcess = null;
        await EnsureReadyAsync(progress, cancellationToken);
    }

    private void StartBackend()
    {
        var logDirectory = Path.Combine(ProjectRoot.FullName, "outputs", "runtime_logs");
        Directory.CreateDirectory(logDirectory);
        lock (_logGate) _logWriter?.Dispose();
        _logWriter = new StreamWriter(Path.Combine(logDirectory, "desktop_supervisor.log"), append: true)
        {
            AutoFlush = true,
        };

        var packagedBackend = Path.Combine(
            ProjectRoot.FullName,
            "bin",
            "UrbanFly.Backend",
            OperatingSystem.IsWindows() ? "UrbanFly.Backend.exe" : "UrbanFly.Backend");
        var python = Environment.GetEnvironmentVariable("URBANFLY_PYTHON") ?? "python";
        var start = new ProcessStartInfo
        {
            FileName = File.Exists(packagedBackend) ? packagedBackend : python,
            WorkingDirectory = ProjectRoot.FullName,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        start.Environment["URBANFLY_ROOT"] = ProjectRoot.FullName;
        start.Environment["PYTHONUNBUFFERED"] = "1";
        if (!File.Exists(packagedBackend))
        {
            start.ArgumentList.Add("-u");
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add(
                "import os,sys; "
                + "log=open(sys.argv[1],'a',buffering=1,encoding='utf-8'); "
                + "os.dup2(log.fileno(),1); os.dup2(log.fileno(),2); "
                + "sys.stdout=log; sys.stderr=log; sys.path.insert(0,'.'); "
                + "from backend.server.server import main; main()");
            start.ArgumentList.Add(Path.Combine(logDirectory, $"desktop_backend_{DateTime.Now:yyyyMMdd_HHmmss}.log"));
        }
        else
        {
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
        }
        _ownedProcess = new Process { StartInfo = start, EnableRaisingEvents = true };
        if (!_ownedProcess.Start())
        {
            throw new InvalidOperationException("Unable to start UrbanFly backend process.");
        }
        if (File.Exists(packagedBackend))
        {
            _ownedProcess.OutputDataReceived += (_, args) => WriteLog(args.Data);
            _ownedProcess.ErrorDataReceived += (_, args) => WriteLog(args.Data);
            _ownedProcess.BeginOutputReadLine();
            _ownedProcess.BeginErrorReadLine();
        }
        WriteLog($"backend pid={_ownedProcess.Id} root={ProjectRoot.FullName}");
    }

    private void WriteLog(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }
        lock (_logGate)
        {
            _logWriter?.WriteLine($"{DateTimeOffset.Now:O} {line}");
        }
    }

    private string LogPath() =>
        Path.Combine(ProjectRoot.FullName, "outputs", "runtime_logs");

    private static DirectoryInfo LocateProjectRoot()
    {
        var configured = Environment.GetEnvironmentVariable("URBANFLY_ROOT");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var candidate = new DirectoryInfo(configured);
            if (IsProjectRoot(candidate))
            {
                return candidate;
            }
            throw new DirectoryNotFoundException(
                $"URBANFLY_ROOT does not contain the UrbanFly backend: {configured}");
        }

        foreach (var origin in new[]
        {
            new DirectoryInfo(AppContext.BaseDirectory),
            new DirectoryInfo(Environment.CurrentDirectory),
        })
        {
            DirectoryInfo? current = origin;
            for (var depth = 0; depth < 10 && current is not null; depth += 1)
            {
                if (IsProjectRoot(current))
                {
                    return current;
                }
                current = current.Parent;
            }
        }
        throw new DirectoryNotFoundException(
            "Cannot locate UrbanFly project root. Set URBANFLY_ROOT explicitly.");
    }

    private static bool IsProjectRoot(DirectoryInfo directory)
    {
        var frontend = File.Exists(Path.Combine(
            directory.FullName, "frontend", "dist", "index.html"));
        var sourceBackend = File.Exists(Path.Combine(
            directory.FullName, "backend", "server", "server.py"));
        var packagedBackend = File.Exists(Path.Combine(
            directory.FullName,
            "bin",
            "UrbanFly.Backend",
            OperatingSystem.IsWindows() ? "UrbanFly.Backend.exe" : "UrbanFly.Backend"));
        var marker = File.Exists(Path.Combine(directory.FullName, "urbanfly-release.json"))
            || File.Exists(Path.Combine(directory.FullName, "AGENTS.md"));
        return marker && frontend && (sourceBackend || packagedBackend);
    }

    public ValueTask DisposeAsync()
    {
        var packagedBackend = Path.Combine(
            ProjectRoot.FullName,
            "bin",
            "UrbanFly.Backend",
            OperatingSystem.IsWindows() ? "UrbanFly.Backend.exe" : "UrbanFly.Backend");
        if (File.Exists(packagedBackend) && _ownedProcess is { HasExited: false })
        {
            // OnClosing already proved that no collector/policy is connected.
            // End the packaged child so uninstall/update never leaves a hidden
            // engine holding files or port 8765. Development backends retain
            // the historical keep-warm behaviour.
            WriteLog("desktop closing; stopping owned packaged backend");
            _ownedProcess.Kill(entireProcessTree: true);
            _ownedProcess.WaitForExit(3000);
        }
        else
        {
            WriteLog("desktop detached; development backend preserved");
        }
        _ownedProcess?.Dispose();
        lock (_logGate)
        {
            _logWriter?.Dispose();
            _logWriter = null;
        }
        _startGate.Dispose();
        _http.Dispose();
        return ValueTask.CompletedTask;
    }
}
