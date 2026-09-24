<#
  Локальный сервер методички. Заменяет "python -m http.server":
  Windows умеет раздавать файлы сам, Python ставить не нужно.

  Запуск:  powershell -NoProfile -ExecutionPolicy Bypass -File serve.ps1
  Параметры: -Port 5500  -NoBrowser
#>

[CmdletBinding()]
param(
    [int]$Port = 5500,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

# Отдаём только то, что лежит рядом с этим скриптом (папка методички).
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$RootFull = [System.IO.Path]::GetFullPath($Root)

$MimeMap = @{
    '.html' = 'text/html; charset=utf-8'
    '.htm'  = 'text/html; charset=utf-8'
    '.js'   = 'text/javascript; charset=utf-8'
    '.mjs'  = 'text/javascript; charset=utf-8'
    '.css'  = 'text/css; charset=utf-8'
    '.json' = 'application/json; charset=utf-8'
    '.map'  = 'application/json; charset=utf-8'
    '.txt'  = 'text/plain; charset=utf-8'
    '.md'   = 'text/plain; charset=utf-8'
    '.svg'  = 'image/svg+xml'
    '.png'  = 'image/png'
    '.jpg'  = 'image/jpeg'
    '.jpeg' = 'image/jpeg'
    '.gif'  = 'image/gif'
    '.webp' = 'image/webp'
    '.ico'  = 'image/x-icon'
    '.bmp'  = 'image/bmp'
    '.woff' = 'font/woff'
    '.woff2' = 'font/woff2'
    '.ttf'  = 'font/ttf'
    '.otf'  = 'font/otf'
    '.wasm' = 'application/wasm'
    '.pdf'  = 'application/pdf'
    '.mp4'  = 'video/mp4'
    '.webmanifest' = 'application/manifest+json'
}

function Write-Line([string]$Text, [string]$Color) {
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}

function Test-PortFree([int]$Candidate) {
    $probe = $null
    try {
        $probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Candidate)
        $probe.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($probe) { $probe.Stop() }
    }
}

function Get-SitePath([System.Net.HttpListenerRequest]$Request) {
    $relative = [System.Uri]::UnescapeDataString($Request.Url.AbsolutePath).TrimStart('/')
    $relative = $relative.Replace([char]'/', [System.IO.Path]::DirectorySeparatorChar)
    if ([string]::IsNullOrWhiteSpace($relative)) { $relative = 'index.html' }

    $full = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($RootFull, $relative))

    # Защита от "../": за пределы папки методички не выходим.
    if (-not $full.StartsWith($RootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }
    if ($full.Equals($RootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        $full = [System.IO.Path]::Combine($RootFull, 'index.html')
    }
    return $full
}

function Send-SimpleResponse([System.Net.HttpListenerContext]$Context, [int]$Code, [string]$Message) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Message)
    $Context.Response.StatusCode = $Code
    $Context.Response.ContentType = 'text/plain; charset=utf-8'
    $Context.Response.ContentLength64 = $bytes.Length
    if ($Context.Request.HttpMethod -ne 'HEAD') {
        $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    }
}

function Send-File([System.Net.HttpListenerContext]$Context, [string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()

    if ($MimeMap.ContainsKey($extension)) {
        $Context.Response.ContentType = $MimeMap[$extension]
    } else {
        $Context.Response.ContentType = 'application/octet-stream'
    }

    # no-cache: поправили методичку — изменения видно сразу, без жёсткого обновления.
    $Context.Response.AddHeader('Cache-Control', 'no-cache')
    $Context.Response.StatusCode = 200
    $Context.Response.ContentLength64 = $bytes.Length

    if ($Context.Request.HttpMethod -ne 'HEAD') {
        $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    }
}

# Порт 5500 может быть занят предыдущим запуском — берём следующий свободный.
$chosenPort = 0
for ($candidate = $Port; $candidate -lt $Port + 25; $candidate++) {
    if (Test-PortFree $candidate) { $chosenPort = $candidate; break }
}
if (-not $chosenPort) {
    Write-Line "  Не нашёл свободный порт в диапазоне $Port...$($Port + 24)." 'Red'
    Write-Line "  Закройте предыдущее окно с сервером и запустите снова."
    exit 1
}

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:$chosenPort/")
try {
    $listener.Start()
} catch {
    Write-Line "  Сервер не смог запуститься: $($_.Exception.Message)" 'Red'
    Write-Line "  Перезагрузите компьютер и попробуйте снова."
    exit 1
}

$url = "http://127.0.0.1:$chosenPort/"

Write-Line ""
Write-Line "  Сайт доступен: $url" 'Green'
Write-Line "  Папка: $RootFull"
Write-Line "  Это окно нельзя закрывать, пока вы на сайте. Остановка — Ctrl+C или закрыть окно."
Write-Line ""

if (-not $NoBrowser) {
    try {
        # Браузер открываем отдельной задачей, чтобы не ждать первой страницы.
        Start-Job -ArgumentList $url -ScriptBlock {
            param($target)
            Start-Sleep -Milliseconds 800
            Start-Process $target
        } | Out-Null
    } catch {
        Write-Line "  Откройте в браузере вручную: $url" 'Yellow'
    }
}

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $shown = $request.Url.AbsolutePath
        if ([string]::IsNullOrWhiteSpace($shown.Trim('/'))) { $shown = '/' }

        try {
            if ($request.HttpMethod -ne 'GET' -and $request.HttpMethod -ne 'HEAD') {
                Send-SimpleResponse $context 405 "Метод $($request.HttpMethod) не поддерживается."
                Write-Line ("  {0}  405  {1}" -f (Get-Date -Format 'HH:mm:ss'), $shown) 'DarkGray'
                continue
            }

            $path = Get-SitePath $request
            if (-not $path) {
                Send-SimpleResponse $context 403 "Доступ запрещён."
                Write-Line ("  {0}  403  {1}" -f (Get-Date -Format 'HH:mm:ss'), $shown) 'DarkYellow'
                continue
            }

            if (Test-Path -LiteralPath $path -PathType Container) {
                $path = [System.IO.Path]::Combine($path, 'index.html')
            }

            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                Send-SimpleResponse $context 404 "Файл не найден: $shown"
                Write-Line ("  {0}  404  {1}" -f (Get-Date -Format 'HH:mm:ss'), $shown) 'DarkYellow'
                continue
            }

            Send-File $context $path
            Write-Line ("  {0}  200  {1}" -f (Get-Date -Format 'HH:mm:ss'), $shown) 'DarkGray'
        } catch {
            Write-Line ("  Ошибка: {0}" -f $_.Exception.Message) 'Red'
            try { Send-SimpleResponse $context 500 "Внутренняя ошибка сервера." } catch { }
        } finally {
            $context.Response.KeepAlive = $false
            $context.Response.Close()
        }
    }
} finally {
    try {
        $listener.Stop()
        $listener.Close()
    } catch { }
    Write-Line ""
    Write-Line "  Сервер остановлен." 'Yellow'
}
