#http module import

import http from 'http';

#http server(req,res을 전달받을 수 있는 곳) 생성

const server = http.createServer((req, res) => {
    /* TODO: 각각의 URL들을 어떻게 처리하면 좋을까요? */
    res.end();
});

server.listen(8000);

#http get

http.get({
  hostname: 'localhost',
  port: 80,
  path: '/',
  agent: false  // Create a new agent just for this one request
}, (res) => {
  // Do stuff with response
});

#error handling

http.get(options, (res) => {
  // Do stuff
}).on('socket', (socket) => {
  socket.emit('agentRemove');
});


#express를 활용하여 REST API 구현(POST)

var express = require('express');
var bodyParser = require('body-parser');
var app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json())

var port = 9000;

app.post('/sample/put/data', function(req, res) {
    console.log('receiving data ...');
    console.log('body is ',req.body);
    res.send(req.body);
});

// start the server
app.listen(port);
console.log('Server started! At http://localhost:' + port);
