const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const axios = require('axios');
const { getEncoding } = require("js-tiktoken");
const fs = require('fs');
const path = require('path');

const port = 8001;

const GPT35TURBO = 'gpt-3.5-turbo';
const GPT4TURBO = 'gpt-4-turbo-preview';

// read system message from prompts/system.txt
const SYSTEM_MESSAGE = fs.readFileSync(path.join(__dirname, 'prompts/system.txt'), 'utf8');

app.use(bodyParser.json());

// read environment variables from .env.json file without dotenv
const env = require('./.env.json');
const enc = getEncoding("cl100k_base");

async function token_count(messages) {
    let text = "";
    for (const message of messages) {
        text += message.content + "\n";
    }
    return enc.encode(text).length;
}

function create_message(role, content) {
    return {
        role: role,
        content: content
    };
}

async function llm(messages, max_tokens=500, temperature=0.7, model='gpt-3.5-turbo') {
    try {
        const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        {
            model: model,
            messages: messages,
            temperature: temperature,
            max_tokens: max_tokens
        },
        {
            headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${env.OPEN_AI_API_KEY}`
            }
        }
        );

        return response.data;
    } catch (error) {
        console.error('Error:', error.response ? error.response.data : error.message);
        throw error;
    }
}

async function respond(messages) {
    // add system message to the beginning of the messages
    messages.unshift(create_message('system', SYSTEM_MESSAGE));
    return llm(messages);
}

// serve static/index.html
app.use(express.static('static'));

app.post('/chat', async (req, res) => {
    const messages = req.body;
    // remove any "system" role messages
    const filtered_messages = messages.filter(m => m.role !== 'system');
    const resp = await respond(filtered_messages);
    res.status(200).json({message: resp.choices[0].message.content});
});


app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});