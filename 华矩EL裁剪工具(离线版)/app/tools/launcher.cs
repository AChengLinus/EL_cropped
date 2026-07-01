// 华矩 EL 裁剪工具（离线版）- 极简启动器
// 编译: C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe /target:winexe /win32icon:app\tools\logo.ico /out:EL裁剪工具-离线版.exe app\tools\launcher.cs

using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;

public class Launcher : Form {
    static string Root { get { return Path.GetDirectoryName(Application.ExecutablePath); } }
    static string PyPath { get { return Path.Combine(Root, "runtime", "python.exe"); } }
    static string AppPath { get { return Path.Combine(Root, "app", "app.py"); } }
    static int Port = 15789;
    static string Url { get { return "http://127.0.0.1:" + Port; } }

    Process proc;
    System.Windows.Forms.Timer statusTimer;

    Label lblStatus, lblUrl;
    Button btnStart, btnStop;

    static bool TestBackend() {
        try { using (var c = new TcpClient()) { var ar = c.BeginConnect("127.0.0.1", Port, null, null); return ar.AsyncWaitHandle.WaitOne(350); } }
        catch { return false; }
    }

    static bool TestFrontend() {
        try { var req = WebRequest.Create(Url + "/health"); req.Timeout = 700; using (var res = req.GetResponse()) using (var sr = new StreamReader(res.GetResponseStream())) return sr.ReadToEnd().IndexOf("frontend") >= 0; }
        catch { return false; }
    }

    public Launcher() {
        Text = "EL裁剪工具 - 离线版";
        Size = new Size(380, 220);
        MinimumSize = new Size(380, 200);
        MaximumSize = new Size(380, 220);
        MaximizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(246, 248, 252);
        Font = new Font("Microsoft YaHei UI", 9);
        FormBorderStyle = FormBorderStyle.FixedSingle;

        try { Icon = new Icon(Path.Combine(Root, "app", "tools", "logo.ico")); } catch { }
        BuildUI();

        statusTimer = new System.Windows.Forms.Timer { Interval = 2000 };
        statusTimer.Tick += (s, e) => {
            bool be = TestBackend();
            if (be) {
                bool fe = TestFrontend();
                if (fe) {
                    SetStatus("running");
                    statusTimer.Stop();
                    Process.Start(Url);
                } else {
                    SetStatus("waiting");
                }
            }
        };

        this.Paint += (s, e) => {
            // 斜角多层分散水印
            var g = e.Graphics;
            var fnt = new Font("Microsoft YaHei UI", 13, FontStyle.Bold);
            var brush = new SolidBrush(Color.FromArgb(12, 0, 0, 0));
            var cw = this.ClientSize.Width;
            var ch = this.ClientSize.Height;
            g.TranslateTransform(cw / 2, ch / 2);
            g.RotateTransform(-22);
            // 多层分散排列
            for (int row = 0; row < 6; row++) {
                int yBase = -250 + row * 80;
                int xOff = (row % 3) * 30; // 每层错开
                for (int x = -350 + xOff; x < 350; x += 140)
                    g.DrawString("华矩检测", fnt, brush, x, yBase);
            }
            g.ResetTransform();
        };

        FormClosing += (s, e) => {
            statusTimer.Stop();
            if (TestBackend()) {
                if (MessageBox.Show("服务仍在运行，是否同时停止？", "关闭", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes) {
                    StopPython();
                    proc = null;
                }
            }
        };
    }

    void BuildUI() {
        var hdr = new Panel { Dock = DockStyle.Top, Height = 52, BackColor = Color.FromArgb(37, 99, 235) };
        MakeLabel(hdr, "华矩 EL 裁剪工具 · 离线版", 20, 8, 340, 24, new Font("Microsoft YaHei UI", 13, FontStyle.Bold), Color.White);
        MakeLabel(hdr, "纯基础算法 · 离线可用 · 即开即用", 22, 30, 340, 16, new Font("Microsoft YaHei UI", 8), Color.FromArgb(219, 234, 254));
        Controls.Add(hdr);

        lblStatus = MakeLabel(this, "就绪，点击启动", 0, 68, 380, 28, new Font("Microsoft YaHei UI", 13, FontStyle.Bold), Color.FromArgb(100, 116, 139));
        lblStatus.TextAlign = ContentAlignment.MiddleCenter;

        // 可点击的地址链接
        lblUrl = new Label {
            Text = Url,
            Location = new Point(0, 96), Size = new Size(380, 24),
            Font = new Font("Microsoft YaHei UI", 10, FontStyle.Underline),
            ForeColor = Color.FromArgb(37, 99, 235),
            TextAlign = ContentAlignment.MiddleCenter,
            Cursor = Cursors.Hand,
            BackColor = Color.Transparent
        };
        lblUrl.Click += (s, e) => Process.Start(Url);
        Controls.Add(lblUrl);

        // 启动按钮
        btnStart = MakeBtn("启动", 30, 138, 140, 40, Color.FromArgb(37, 99, 235), Color.White, () => StartService());
        btnStart.Font = new Font("Microsoft YaHei UI", 13, FontStyle.Bold);

        // 停止按钮
        btnStop = MakeBtn("停止服务", 180, 138, 140, 40, Color.FromArgb(220, 38, 38), Color.White, () => StopService());
        btnStop.Font = new Font("Microsoft YaHei UI", 13, FontStyle.Bold);
        btnStop.Enabled = false;

    }

    Label MakeLabel(Control parent, string t, int x, int y, int w, int h, Font f, Color c) {
        var l = new Label { Text = t, Location = new Point(x, y), Size = new Size(w, h), Font = f, ForeColor = c };
        if (parent != null) l.BackColor = (parent.BackColor == Color.Empty || parent.BackColor == Color.Transparent) ? Color.Transparent : parent.BackColor;
        parent.Controls.Add(l); return l;
    }

    Button MakeBtn(string t, int x, int y, int w, int h, Color bg, Color fg, Action act) {
        var b = new Button { Text = t, Location = new Point(x, y), Size = new Size(w, h), BackColor = bg, ForeColor = fg, FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand };
        b.FlatAppearance.BorderSize = 0; b.Click += (s, e) => act(); Controls.Add(b); return b;
    }

    void SetStatus(string state) {
        if (InvokeRequired) { BeginInvoke(new Action(() => SetStatus(state))); return; }
        switch (state) {
            case "running":
                lblStatus.Text = "● 服务运行中";
                lblStatus.ForeColor = Color.FromArgb(22, 163, 74);
                lblUrl.Text = Url;
                btnStart.Enabled = false;
                btnStart.Text = "已启动";
                btnStart.BackColor = Color.FromArgb(22, 163, 74);
                btnStop.Enabled = true;
                break;
            case "waiting":
                lblStatus.Text = "● 后端启动中，等待就绪...";
                lblStatus.ForeColor = Color.FromArgb(217, 119, 6);
                lblUrl.Text = "";
                btnStop.Enabled = true;
                break;
        }
    }

    void StartService() {
        if (!File.Exists(PyPath)) { lblStatus.Text = "错误：未找到 Python 运行环境"; lblStatus.ForeColor = Color.FromArgb(220, 38, 38); return; }
        if (!File.Exists(AppPath)) { lblStatus.Text = "错误：未找到后端入口 app.py"; lblStatus.ForeColor = Color.FromArgb(220, 38, 38); return; }
        if (TestBackend()) {
            SetStatus("running");
            return;
        }
        lblStatus.Text = "● 正在启动服务...";
        lblStatus.ForeColor = Color.FromArgb(217, 119, 6);
        btnStart.Enabled = false;
        btnStart.Text = "启动中...";
        btnStop.Enabled = true;

        var psi = new ProcessStartInfo {
            FileName = PyPath, Arguments = "\"" + AppPath + "\"",
            WorkingDirectory = Path.GetDirectoryName(AppPath),
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = false, RedirectStandardError = false,
        };
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["EL_CROP_PORT"] = Port.ToString();
        proc = new Process { StartInfo = psi };
        proc.Start();

        statusTimer.Start();
    }

    void StopService() {
        statusTimer.Stop();
        StopPython();
        proc = null;
        lblStatus.Text = "就绪，点击启动";
        lblStatus.ForeColor = Color.FromArgb(100, 116, 139);
        lblUrl.Text = Url;
        btnStart.Enabled = true;
        btnStart.Text = "启动";
        btnStart.BackColor = Color.FromArgb(37, 99, 235);
        btnStop.Enabled = false;
    }

    void StopPython() {
        try {
            var p = Process.Start(new ProcessStartInfo("netstat", "-ano -p tcp") { UseShellExecute = false, RedirectStandardOutput = true, CreateNoWindow = true });
            foreach (var line in p.StandardOutput.ReadToEnd().Split('\n')) {
                if (line.Contains(":" + Port) && line.Contains("LISTENING")) {
                    var parts = line.Trim().Split(' ');
                    int pid;
                    if (int.TryParse(parts[parts.Length - 1], out pid) && pid != Process.GetCurrentProcess().Id) {
                        Process.Start("taskkill", "/F /PID " + pid).WaitForExit(3000);
                    }
                }
            }
        } catch { }
    }

    [STAThread]
    static void Main() {
        bool createdNew;
        using (var mutex = new System.Threading.Mutex(true, @"Global\EL-Crop-Tool-Offline", out createdNew)) {
            if (!createdNew) {
                var existing = Process.GetProcessesByName("EL裁剪工具-离线版");
                if (existing.Length > 0) {
                    SetForegroundWindow(existing[0].MainWindowHandle);
                }
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Launcher());
        }
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    static extern bool SetForegroundWindow(IntPtr hWnd);
}
