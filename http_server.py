#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
语音识别API的HTTP服务器程序

"""
import http.server
import os
import cgi
import json
import api_python

ds = api_python.loadModel()

# 创建上传文件夹
upload_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "uploads"))
folder = os.path.exists(upload_dir)
if not folder:
    os.makedirs(upload_dir)

class HTTPHandle(http.server.BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(100)
        http.server.BaseHTTPRequestHandler.setup(self)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()


    def do_POST(self):
        if(self.path != '/asr'):
            print('----------------------------非法请求：'+self.path)
            self._set_response()
            buf = bytes('403', encoding="utf-8")
            self.wfile.write(buf)
            return
        '''
        处理通过POST方式传递过来并接收的语音数据
        通过语音模型和语言模型计算得到语音识别结果并返回
        '''
        

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     }
        )

        field_item = form['name']
        filename = field_item.filename
        filevalue = field_item.value
        final_path = os.path.join(upload_dir, filename)
        print('------------save file:'+final_path)
        # file_length = len(file_data['body'])
        output_file = open(final_path, 'wb')
        output_file.write(filevalue)
        output_file.close()
        print('------------save file ok')

        # 提取文件特征
        pinyin = api_python.getWord(ds, final_path)
        res = {'code': 99999, 'data': {
            'pinyin': pinyin, 'hanzi': '', 'score': 80}}

        resultJson = json.dumps(res, ensure_ascii=False)
        self._set_response()
        buf = bytes(resultJson, encoding="utf-8")
        self.wfile.write(buf)
        print('------------response ：'+resultJson)


def start_server(ip, port):
    http_server = http.server.HTTPServer((ip, int(port)), HTTPHandle)
    print('服务器已开启')

    try:
        http_server.serve_forever()  # 设置一直监听并接收请求
    except KeyboardInterrupt:
        pass
    http_server.server_close()
    print('HTTP server closed')


if __name__ == '__main__':
    start_server('', 9527)
