import { getProjectReply } from './project-replies.js';

const $ = (id) => document.getElementById(id);
const form = $('chat-form');
const input = $('message-input');
const send = $('send-button');
const log = $('messages');
const history = [];
const modelPicker = $('model-picker');
const modelTrigger = $('model-select');
const modelMenu = $('model-menu');
const currentModelOption = $('model-100k');
let model = null;
let busy = false;
let generationId = 0;
let loadFailed = false;

function closeModelMenu(returnFocus = false) {
  modelMenu.hidden = true;
  document.body.classList.remove('model-menu-open');
  modelTrigger.setAttribute('aria-expanded', 'false');
  if (returnFocus && !modelTrigger.disabled) modelTrigger.focus();
}

function openModelMenu() {
  if (busy) return;
  modelMenu.hidden = false;
  document.body.classList.add('model-menu-open');
  modelTrigger.setAttribute('aria-expanded', 'true');
  currentModelOption.focus();
}

modelTrigger.addEventListener('click', () => {
  if (modelMenu.hidden) openModelMenu();
  else closeModelMenu(true);
});
modelTrigger.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    openModelMenu();
  }
});
currentModelOption.addEventListener('click', () => closeModelMenu(true));
modelMenu.addEventListener('keydown', (event) => {
  if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
    event.preventDefault();
    currentModelOption.focus();
  }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !modelMenu.hidden) {
    event.preventDefault();
    closeModelMenu(true);
  }
});
document.addEventListener('pointerdown', (event) => {
  if (!modelPicker.contains(event.target)) closeModelMenu();
});
document.addEventListener('focusin', (event) => {
  if (!modelPicker.contains(event.target)) closeModelMenu();
});

function updateComposer() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  send.disabled = busy || !input.value.trim() || (!model && !getProjectReply(input.value));
  $('model-select').disabled = busy;
  if (busy) closeModelMenu();
}

function appendMessage(content, kind) {
  const item = document.createElement('p');
  item.className = `message message-${kind}`;
  item.textContent = content;
  log.append(item);
  log.scrollTop = log.scrollHeight;
  return item;
}

function createAnswerStream(item, currentId) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let target = [];
  let visible = 0;
  let pending;
  let stopped = false;
  item.classList.add('is-streaming');

  const render = () => {
    item.textContent = target.slice(0, visible).join('');
    log.scrollTop = log.scrollHeight;
  };
  const update = (text) => {
    if (stopped || currentId !== generationId) return;
    target = Array.from(text);
    if (reducedMotion) {
      visible = target.length;
      render();
      return;
    }
    pending ??= (async () => {
      while (!stopped && currentId === generationId && visible < target.length) {
        await new Promise((resolve) => setTimeout(resolve, 20));
        if (stopped || currentId !== generationId) break;
        visible = Math.min(visible + 2, target.length);
        render();
      }
    })().finally(() => { pending = undefined; });
  };

  return {
    update,
    async finish(text) {
      update(text);
      await pending;
    },
    stop() {
      stopped = true;
      item.classList.remove('is-streaming');
    },
  };
}

function setChatStarted(started) {
  closeModelMenu();
  $('privacy-note').hidden = started || location.hash === '#about';
  $('welcome').hidden = started;
  $('suggestions').hidden = started;
  $('conversation-header').hidden = !started;
  log.hidden = !started;
  document.body.classList.toggle('chat-active', started && location.hash !== '#about');
}

function route() {
  closeModelMenu();
  const about = location.hash === '#about';
  $('chat-page').hidden = about;
  $('about-page').hidden = !about;
  $('privacy-note').hidden = about || !log.hidden;
  document.body.classList.toggle('chat-active', !about && !log.hidden);
  if (about) $('about-link').setAttribute('aria-current', 'page');
  else $('about-link').removeAttribute('aria-current');
  document.title = about ? 'about | PicoBoi' : 'PicoBoi';
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

  const projectReply = getProjectReply(content);
  if (projectReply) {
    const item = appendMessage('', 'assistant');
    const link = document.createElement('a');
    link.href = projectReply.url;
    link.textContent = projectReply.label;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    item.append(projectReply.prefix, link, projectReply.suffix);
    item.dataset.responseSource = 'project';
    log.scrollTop = log.scrollHeight;
    if (!$('chat-page').hidden) input.focus();
    return;
  }
  if (!model) return;

  const currentId = ++generationId;
  history.push({ role: 'user', content });
  busy = true;
  log.setAttribute('aria-busy', 'true');
  $('model-status').textContent = 'Thinking…';
  updateComposer();
  let answerElement;
  let answerStream;
  const streamAnswer = (text) => {
    if (currentId !== generationId) return;
    answerElement ??= appendMessage('', 'assistant');
    answerStream ??= createAnswerStream(answerElement, currentId);
    $('model-status').textContent = 'Replying…';
    answerStream.update(text);
  };
  try {
    const result = await model.generate(history.map((message) => ({ ...message })), streamAnswer, (text) => {
      if (currentId === generationId) appendMessage(`I read that as: ${text}`, 'notice');
    });
    const response = result.text;
    if (currentId !== generationId) return;
    if (typeof response !== 'string' || !response.trim()) throw new Error('Empty model response');
    streamAnswer(response);
    await answerStream.finish(response);
    if (currentId !== generationId) return;
    history[history.length - 1].content = result.normalizedInput;
    history.push({ role: 'assistant', content: response });
    if (result.dropped) appendMessage('Older messages were left out to fit the 128-token context window.', 'notice');
    if (result.truncated) appendMessage('This answer reached the model’s length limit.', 'notice');
  } catch (error) {
    if (currentId !== generationId) return;
    history.pop();
    answerElement?.remove();
    appendMessage(error.message || 'Something interrupted the response. Please try again.', 'notice');
  } finally {
    answerStream?.stop();
    if (currentId === generationId) {
      busy = false;
      log.removeAttribute('aria-busy');
      $('model-status').textContent = model ? 'On device' : loadFailed ? 'Load failed' : 'Loading…';
      updateComposer();
      if (!$('chat-page').hidden) input.focus();
    }
  }
});

$('new-chat').addEventListener('click', () => {
  generationId++;
  model?.cancel?.();
  busy = false;
  log.removeAttribute('aria-busy');
  history.length = 0;
  log.replaceChildren();
  input.value = '';
  setChatStarted(false);
  $('model-status').textContent = model ? 'On device' : loadFailed ? 'Load failed' : 'Loading…';
  updateComposer();
  input.focus();
});

function connectLocalModel() {
  const worker = new Worker(new URL('./model-worker.js?v=normalizer1', import.meta.url), { type: 'module' });
  let sequence = 0;
  let pending;
  const adapter = {
    generate(messages, onToken, onNormalized) {
      if (pending) return Promise.reject(new Error('A response is already running.'));
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        pending = { id, resolve, reject, onToken, onNormalized };
        worker.postMessage({ type: 'generate', id, messages });
      });
    },
    cancel() {
      worker.postMessage({ type: 'cancel', id: ++sequence });
      pending?.reject(new Error('Cancelled'));
      pending = undefined;
    },
  };
  const failed = (message) => {
    loadFailed = true;
    model = null;
    $('model-status').textContent = 'Load failed';
    pending?.reject(new Error(message));
    pending = undefined;
    $('model-help').textContent = message;
    updateComposer();
  };
  worker.addEventListener('error', () => failed('The local model could not start. Please reload the page.'));
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      model = adapter;
      $('model-status').textContent = 'On device';
      updateComposer();
      return;
    }
    if (data.type === 'load-error') { failed(data.message); return; }
    if (!pending || data.id !== pending.id) return;
    if (data.type === 'normalized') { pending.onNormalized(data.text); return; }
    if (data.type === 'token') { pending.onToken(data.text); return; }
    const request = pending;
    pending = undefined;
    if (data.type === 'done') request.resolve(data);
    else request.reject(new Error(data.message));
  });
}

try { connectLocalModel(); }
catch { loadFailed = true; $('model-status').textContent = 'Load failed'; $('model-help').textContent = 'This browser could not start the local model. Try a current browser.'; }

window.addEventListener('hashchange', route);
route();
updateComposer();
