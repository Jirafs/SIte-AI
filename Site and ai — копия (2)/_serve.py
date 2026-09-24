import os
import http.server

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Site and ai"))
http.server.test(
    HandlerClass=http.server.SimpleHTTPRequestHandler,
    port=5500,
    bind="127.0.0.1",
)
