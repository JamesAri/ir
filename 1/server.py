# To test the retry mechanism
from http.server import BaseHTTPRequestHandler, HTTPServer


class TestRetryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        status_code = 503
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        print(f"Returned status code: {status_code}")


def run(server_class=HTTPServer, handler_class=TestRetryHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting local server at http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
