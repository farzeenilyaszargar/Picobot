// Attach a local inference engine later with:
// window.picoboi.setModel({ async generate(messages) { return 'model output'; } });
// `messages` contains { role: 'user' | 'assistant', content: string } objects.
// No remote service or fabricated model response is used in this frontend.
const $ = (id) => document.getElementById(id);
const form = $('chat-form');
const input = $('message-input');
const send = $('send-button');
const log = $('messages');
const history = [];
let model = null;
let busy = false;
let generationId = 0;

function updateComposer() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  send.disabled = busy || !input.value.trim();
  $('model-select').disabled = busy;
}

function appendMessage(content, kind) {
  const item = document.createElement('p');
  item.className = `message message-${kind}`;
  item.textContent = content;
  log.append(item);
  log.scrollTop = log.scrollHeight;
}

function setChatStarted(started) {
  $('welcome').hidden = started;
  $('suggestions').hidden = started;
  $('conversation-header').hidden = !started;
  log.hidden = !started;
  document.body.classList.toggle('chat-active', started && location.hash !== '#about');
}

function route() {
  const about = location.hash === '#about';
  $('chat-page').hidden = about;
  $('about-page').hidden = !about;
  document.body.classList.toggle('chat-active', !about && !log.hidden);
  if (about) $('about-link').setAttribute('aria-current', 'page');
  else $('about-link').removeAttribute('aria-current');
  document.title = about ? 'about | picoboi' : 'picoboi';
  if (about) $('about-heading').focus({ preventScroll: true });
}

input.addEventListener('input', updateComposer);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    updateComposer();
    input.focus();
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || busy) return;
  setChatStarted(true);
  appendMessage(content, 'user');
  input.value = '';
  updateComposer();

  if (!model) {
    appendMessage('Picoboi is still taking shape. The language model isn’t connected yet. This is a preview of its future home.', 'notice');
    input.focus();
    return;
  }

  const currentId = ++generationId;
  history.push({ role: 'user', content });
  busy = true;
  log.setAttribute('aria-busy', 'true');
  $('model-status').textContent = 'Thinking…';
  updateComposer();
  try {
    const response = await model.generate(history.map((message) => ({ ...message })));
    if (currentId !== generationId) return;
    if (typeof response !== 'string' || !response.trim()) throw new Error('Empty model response');
    history.push({ role: 'assistant', content: response });
    appendMessage(response, 'assistant');
  } catch {
    if (currentId !== generationId) return;
    history.pop();
    appendMessage('Something interrupted the response. Please try sending your message again.', 'notice');
  } finally {
    if (currentId === generationId) {
      busy = false;
      log.removeAttribute('aria-busy');
      $('model-status').textContent = model ? 'Connected' : 'Not connected';
      updateComposer();
      if (!$('chat-page').hidden) input.focus();
    }
  }
});

$('new-chat').addEventListener('click', () => {
  generationId++;
  busy = false;
  log.removeAttribute('aria-busy');
  history.length = 0;
  log.replaceChildren();
  input.value = '';
  setChatStarted(false);
  $('model-status').textContent = model ? 'Connected' : 'Not connected';
  updateComposer();
  input.focus();
});

window.picoboi = Object.freeze({
  setModel(adapter) {
    if (busy) throw new Error('Finish the current response before switching models.');
    if (adapter !== null && typeof adapter?.generate !== 'function') {
      throw new TypeError('A model adapter must provide an async generate(messages) function.');
    }
    model = adapter;
    history.length = 0;
    $('model-status').textContent = model ? 'Connected' : 'Not connected';
  },
});

window.addEventListener('hashchange', route);
route();
updateComposer();
