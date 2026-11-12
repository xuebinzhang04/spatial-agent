"""Server restart and video-seeking contracts without extra listening ports."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run import handler,stop_previous


class ServerTests(unittest.TestCase):
    def serve_file(self,path,range_header=None):
        request=object.__new__(handler(None,lambda:None))
        request.path='/static/media/clip.mp4'
        request.headers={'Range':range_header} if range_header else {}
        request.request_version='HTTP/1.1';request.command='GET'
        request.wfile=io.BytesIO();request.log_request=lambda *args:None
        request.translate_path=lambda _:str(path)
        file=request.send_head()
        if file:
            with file:request.copyfile(file,request.wfile)
        return request.wfile.getvalue().split(b'\r\n\r\n',1)

    def test_video_ranges_and_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'clip.mp4';path.write_bytes(b'0123456789abcdefghij')
            for header,body in [('bytes=5-9',b'56789'),('bytes=15-',b'fghij'),('bytes=-3',b'hij')]:
                headers,actual=self.serve_file(path,header)
                self.assertIn(b'206 Partial Content',headers)
                self.assertIn(b'Accept-Ranges: bytes',headers)
                self.assertEqual(actual,body)
            for header in ('bytes=100-','bytes=10-2','bytes=-0','bytes=0-1,3-4'):
                headers,body=self.serve_file(path,header)
                self.assertIn(b'416',headers);self.assertEqual(body,b'')
            headers,body=self.serve_file(path)
            self.assertIn(b'200 OK',headers);self.assertEqual(body,path.read_bytes())
            self.assertIn(b'immutable',headers)

    def test_client_assets_disable_stale_caching(self):
        request=object.__new__(handler(None,lambda:None))
        request.path='/app.js';request.wfile=io.BytesIO();request._headers_buffer=[]
        request.request_version='HTTP/1.1'
        request.end_headers()
        self.assertIn(b'Cache-Control: no-store',request.wfile.getvalue())

    def test_restart_refuses_an_unrelated_same_user_process(self):
        listing=SimpleNamespace(stdout=f'LISTEN users:(("python",pid={os.getpid()},fd=3))')
        with patch('run.subprocess.run',return_value=listing),patch('run.os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'another service'):stop_previous(8768)
            kill.assert_not_called()



if __name__=='__main__':unittest.main(verbosity=2)
