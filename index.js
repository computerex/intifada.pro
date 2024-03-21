const express = require('express');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.json());

const port = 8001;

// serve static/index.html
app.use(express.static('static'));

app.get('/hello', (req, res) => {
    res.status(200).send('hello world');
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});