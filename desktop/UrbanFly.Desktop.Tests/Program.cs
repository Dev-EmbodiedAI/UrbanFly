using System.Net;
using System.Net.Sockets;
using System.Text;
using UrbanFly.Desktop;

var socket = new TcpListener(IPAddress.Loopback, 0);
socket.Start();
var port = ((IPEndPoint)socket.LocalEndpoint).Port;
socket.Stop();
var endpoint = new Uri($"http://127.0.0.1:{port}/");
using var listener = new HttpListener();
listener.Prefixes.Add(endpoint.AbsoluteUri);
listener.Start();
await using var supervisor = new BackendSupervisor(new DirectoryInfo(Environment.CurrentDirectory), endpoint);
var cases = new[] {
    ("{\"schema\":\"urbanfly-runtime-health-v1\",\"status\":\"ok\",\"simulator\":{\"state\":\"paused\"},\"clients\":{\"policy\":1,\"total\":2}}", true, (int?)1),
    ("{}", false, (int?)null),
    ("{\"schema\":\"unknown\"}", false, (int?)null),
    ("not-json", false, (int?)null),
    ("[]", false, (int?)null),
};
foreach (var (payload, healthy, policy) in cases)
{
    var read = supervisor.GetHealthAsync();
    var context = await listener.GetContextAsync();
    var bytes = Encoding.UTF8.GetBytes(payload);
    context.Response.ContentLength64 = bytes.Length;
    await context.Response.OutputStream.WriteAsync(bytes);
    context.Response.Close();
    var result = await read;
    if (result.Healthy != healthy || result.PolicyClients != policy) throw new Exception($"Unexpected health: {result}");
}
listener.Stop();
var offline = await supervisor.GetHealthAsync();
if (offline.Healthy || offline.PolicyClients is not null) throw new Exception("Offline must not mean idle.");
Console.WriteLine("PASS: 6 desktop health parsing/offline safety cases");
