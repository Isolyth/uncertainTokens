INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uncertain Tokens</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
#chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.8;
}
.msg.user {
    align-self: flex-end;
    background: #1a3a5c;
    color: #d0e8ff;
}
.msg.assistant {
    align-self: flex-start;
    background: #1a1a2e;
}
#input-bar {
    display: flex;
    padding: 12px 20px;
    gap: 10px;
    background: #111118;
    border-top: 1px solid #222;
}
#input-bar input {
    flex: 1;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #333;
    background: #1a1a2e;
    color: #e0e0e0;
    font-size: 15px;
    outline: none;
}
#input-bar input:focus { border-color: #556; }
#input-bar button {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    background: #2a4a6c;
    color: #d0e8ff;
    font-size: 15px;
    cursor: pointer;
}
#input-bar button:hover { background: #3a5a7c; }
#input-bar button:disabled { opacity: 0.5; cursor: default; }

/* Tokens */
.tok {
    cursor: pointer;
    border-radius: 3px;
    padding: 1px 0;
    white-space: pre-wrap;
}
.tok.mix {
    border-bottom: 1px dashed rgba(150, 100, 255, 0.5);
}
.tok.multimix {
    border-bottom: 2px solid rgba(255, 50, 150, 0.7);
}

/* Tooltip */
#tooltip {
    display: none;
    position: fixed;
    background: #12121f;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 12px;
    font-family: monospace;
    white-space: nowrap;
    z-index: 10000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    pointer-events: none;
}
#tooltip table { border-collapse: collapse; }
#tooltip td { padding: 1px 6px; }
#tooltip .tok-name { text-align: left; color: #8cf; }
#tooltip .tok-prob { text-align: right; color: #fc8; }
#tooltip tr:first-child td { font-weight: bold; }
</style>
</head>
<body>
<div id="chat"></div>
<div id="input-bar">
    <input id="msg" type="text" placeholder="Type a message..." autofocus>
    <button id="send">Send</button>
    <button id="stop" style="display:none;background:#6c2a2a;color:#ffd0d0">Stop</button>
</div>
<div id="tooltip"></div>
<script>
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop');
const tooltip = document.getElementById('tooltip');
let history = [];
let abortController = null;

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (typeof content === 'string') {
        div.textContent = content;
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
}

async function send() {
    const text = msgInput.value.trim();
    if (!text) return;
    msgInput.value = '';
    sendBtn.style.display = 'none';
    stopBtn.style.display = '';

    history.push({role: 'user', content: text});
    addMessage('user', text);

    const assistantDiv = addMessage('assistant', '');
    let fullText = '';

    abortController = new AbortController();

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages: history}),
            signal: abortController.signal,
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});

            const lines = buffer.split('\\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);
                if (payload === '[DONE]') continue;

                const data = JSON.parse(payload);
                fullText += data.token;

                const span = document.createElement('span');
                const isMultimix = data.multimix || false;
                const isMix = data.mix;
                span.className = 'tok' + (isMultimix ? ' multimix' : (isMix ? ' mix' : ''));

                let color;
                if (isMultimix) {
                    color = `rgba(255, 50, 150, ${Math.min(data.uncertainty * 1.5, 0.9)})`;
                } else if (isMix) {
                    color = `rgba(150, 100, 255, ${data.uncertainty})`;
                } else {
                    color = `rgba(255, 150, 0, ${data.uncertainty})`;
                }
                span.style.background = color;
                span.textContent = data.token;
                span.dataset.top = JSON.stringify(data.top);
                span.dataset.mix = isMix ? '1' : '';
                span.dataset.multimix = isMultimix ? '1' : '';
                if (data.sequence_length) {
                    span.dataset.seqLen = data.sequence_length;
                }
                assistantDiv.appendChild(span);
                chat.scrollTop = chat.scrollHeight;
            }
        }
    } catch (e) {
        if (e.name !== 'AbortError') {
            assistantDiv.textContent = 'Error: ' + e.message;
        }
    }

    abortController = null;
    history.push({role: 'assistant', content: fullText});
    sendBtn.style.display = '';
    stopBtn.style.display = 'none';
    msgInput.focus();
}

sendBtn.addEventListener('click', send);
stopBtn.addEventListener('click', () => { if (abortController) abortController.abort(); });
msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

// Tooltip
document.addEventListener('mouseover', e => {
    const tok = e.target.closest('.tok');
    if (!tok || !tok.dataset.top) return;

    const top = JSON.parse(tok.dataset.top);
    const isMix = tok.dataset.mix === '1';
    const isMultimix = tok.dataset.multimix === '1';
    let label = '';
    if (isMultimix) {
        const seqLen = tok.dataset.seqLen || '?';
        label = '<div style="color:#ff3296;font-weight:bold;margin-bottom:4px">MULTI-MIX (' + seqLen + ' tokens collapsed)</div>';
    } else if (isMix) {
        label = '<div style="color:#b88cff;font-weight:bold;margin-bottom:4px">MIX TOKEN (weighted blend)</div>';
    }
    let rows = top.map(t =>
        `<tr><td class="tok-name">${escapeHtml(t.token)}</td><td class="tok-prob">${(t.prob * 100).toFixed(1)}%</td></tr>`
    ).join('');
    tooltip.innerHTML = label + '<table>' + rows + '</table>';
    tooltip.style.display = 'block';

    const rect = tok.getBoundingClientRect();
    let y = rect.bottom + 4;
    let x = rect.left;
    if (y + tooltip.offsetHeight > window.innerHeight) {
        y = rect.top - tooltip.offsetHeight - 4;
    }
    if (x + tooltip.offsetWidth > window.innerWidth) {
        x = window.innerWidth - tooltip.offsetWidth - 8;
    }
    tooltip.style.top = y + 'px';
    tooltip.style.left = x + 'px';
});

document.addEventListener('mouseout', e => {
    if (e.target.closest('.tok')) {
        tooltip.style.display = 'none';
    }
});
</script>
</body>
</html>
"""
