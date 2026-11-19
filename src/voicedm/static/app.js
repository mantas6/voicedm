"use strict";

// voicedm page: sessions, their settings, and a call over one RTCPeerConnection.
// Server events arrive as RTVI messages on the WebRTC data channel.

const $ = (id) => document.getElementById(id);
const els = {
  sidebar: $("sidebar"),
  scrim: $("scrim"),
  menu: $("menu"),
  sessions: $("sessions"),
  newSession: $("new-session"),
  title: $("title"),
  settings: $("settings"),
  settingsSummary: $("settings-summary"),
  model: $("model"),
  voice: $("voice"),
  prompt: $("system-prompt"),
  savePrompt: $("save-prompt"),
  settingsNote: $("settings-note"),
  transcript: $("transcript"),
  status: $("status"),
  dot: $("state-dot"),
  mute: $("mute"),
  call: $("call"),
  remote: $("remote"),
  notices: $("notices"),
  confirm: $("confirm"),
  confirmText: $("confirm-text"),
};

// Pause after "bot stopped speaking" before showing "Listening": the server reports
// a stop between sentences whenever speech synthesis is slower than playback.
const LISTEN_DELAY_MS = 700;
// Time for the server to save the last turn after a hang-up before reloading.
const RELOAD_DELAY_MS = 1200;
// After the connection reports "disconnected", how long to wait for it to recover
// on its own before treating the call as dropped.
const DISCONNECT_GRACE_MS = 3000;
// Pauses before each reconnect attempt after a dropped call.
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000, 15000];
// How often to check the upstreams while not in a call (and when the page regains
// focus); more often while the language model server is down.
const HEALTH_POLL_MS = 30000;
const HEALTH_POLL_DOWN_MS = 10000;

const state = {
  session: null, // current session, with messages
  sessions: [],
  models: [],
  voices: [],
  pc: null,
  mic: null,
  muted: false,
  callState: "idle",
  listenTimer: null,
  live: null, // the live transcript of the current call
  // The current call: { pc, channel, pcId, sessionId, ended }. `pc` is also kept in
  // state.pc; while reconnecting after a drop, state.pc is null and state.call stays.
  call: null,
  reconnect: null, // { attempt, timer } while reconnecting after a dropped call
  warming: null, // { since, pending: [...] } while upstreams warm up
  warmTimer: null,
  health: null, // last /api/health answer
  healthTimer: null,
  notices: new Map(), // key -> { kind, text }
};

// -- helpers --

async function api(method, path, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail) ? detail[0]?.msg : (detail?.message ?? detail);
    const err = new Error(message ?? `HTTP ${response.status}`);
    err.status = response.status;
    err.detail = detail;
    throw err;
  }
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function shortModel(name) {
  // "registry.example/org/some-model:tag" -> "some-model:tag"
  return name.split("/").pop();
}

function shortVoice(name) {
  return name.replace(/\.(wav|mp3|flac|ogg)$/i, "");
}

function relativeTime(iso) {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

// In a call: connected, connecting, or reconnecting after a drop.
function inCall() {
  return state.call !== null;
}

// -- status --

const STATUS_TEXT = {
  idle: "Idle",
  connecting: "Connecting…",
  warming: "Warming up…",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
  reconnecting: "Reconnecting…",
};

const UPSTREAM_NAMES = { ollama: "the language model", chatterbox: "the voice" };

function setCallState(name, text) {
  clearTimeout(state.listenTimer);
  state.listenTimer = null;
  state.callState = name;
  state.statusText = text ?? null;
  els.dot.dataset.state = name;
  renderStatus();
}

// The status line; while upstreams warm up (which can take tens of seconds on a cold
// start), it says what is still loading and for how long.
function renderStatus() {
  const name = state.callState;
  let text = state.statusText ?? STATUS_TEXT[name] ?? name;
  const warming = state.warming;
  if (warming && ["warming", "listening", "thinking"].includes(name)) {
    const seconds = Math.round((Date.now() - warming.since) / 1000);
    const what = warming.pending.map((p) => UPSTREAM_NAMES[p] ?? p).join(" and ");
    if (name === "warming") text = `Warming up ${what}… ${seconds}s`;
    else text += ` · warming up ${what} (${seconds}s)`;
  }
  els.status.textContent = text;
}

function setWarming(pending) {
  if (pending.length) {
    state.warming = { since: state.warming?.since ?? Date.now(), pending };
    state.warmTimer ??= setInterval(renderStatus, 1000);
  } else {
    state.warming = null;
    clearInterval(state.warmTimer);
    state.warmTimer = null;
  }
  renderStatus();
}

// -- notices: upstream problems and call errors, shown above the call controls --

function setNotice(key, kind, text) {
  if (text) state.notices.set(key, { kind, text });
  else state.notices.delete(key);
  renderNotices();
}

// Notices that say less than another one shown at the same time.
const COVERED_BY = {
  models: ["ollama"],
  voices: ["chatterbox"],
  busy: ["call"],
  chatterbox: ["call-tts", "warmup", "call"],
};

function renderNotices() {
  els.notices.replaceChildren();
  const shown = new Set();
  for (const [key, { kind, text }] of state.notices) {
    const covered = (COVERED_BY[key] ?? []).some((other) => state.notices.has(other));
    if (covered || shown.has(text)) continue;
    shown.add(text);
    const item = el("li", `notice ${kind}`);
    const close = el("button", null, "×");
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss");
    close.addEventListener("click", () => setNotice(key));
    item.append(el("p", null, text), close);
    els.notices.append(item);
  }
}

// -- sessions --

const TRASH_ICON =
  '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" ' +
  'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-12M9 7V4h6v3"/>' +
  "</svg>";

function messageCount(session) {
  return session.message_count === 1 ? "1 message" : `${session.message_count} messages`;
}

function renderSessions() {
  els.sessions.replaceChildren();
  for (const session of state.sessions) {
    const button = el("button", "session-select");
    button.type = "button";
    const meta = `${messageCount(session)} · ${relativeTime(session.updated_at)}`;
    button.append(
      el("span", "session-title", session.title || "Untitled session"),
      el("span", "session-meta", session.in_call ? `${meta} · in a call` : meta),
    );
    const current = state.session?.id === session.id;
    button.setAttribute("aria-current", String(current));
    button.disabled = inCall() && !current;
    button.addEventListener("click", () => {
      closeDrawer();
      if (!current) selectSession(session.id);
    });
    const remove = el("button", "session-delete");
    remove.type = "button";
    remove.innerHTML = TRASH_ICON;
    remove.title = "Delete session";
    remove.setAttribute("aria-label", `Delete session “${session.title || "Untitled session"}”`);
    remove.disabled = inCall();
    remove.addEventListener("click", () => deleteSession(session));
    const item = el("li", current ? "current" : "");
    item.append(button, remove);
    els.sessions.append(item);
  }
  els.newSession.disabled = inCall();
}

function sessionTitle(id) {
  const session = state.sessions.find((s) => s.id === id);
  return session?.title || "an untitled session";
}

async function loadSessions() {
  state.sessions = await api("GET", "/api/sessions");
  renderSessions();
}

async function selectSession(id) {
  const session = await api("GET", `/api/sessions/${id}`);
  state.session = session;
  history.replaceState(null, "", `#${session.id}`);
  renderSession();
  renderSessions();
}

async function newSession() {
  try {
    const body = state.session ? { copy_from: state.session.id } : {};
    const session = await api("POST", "/api/sessions", body);
    await loadSessions();
    await selectSession(session.id);
    closeDrawer();
  } catch (err) {
    setCallState("error", `New session failed: ${err.message}`);
  }
}

// Ask before deleting; resolves to true only on an explicit "Delete".
function confirmDelete(session) {
  const name = `“${session.title || "Untitled session"}”`;
  const what = session.message_count ? `${name} and its ${messageCount(session)}` : name;
  const text = `${what} will be deleted for good. This cannot be undone.`;
  if (typeof els.confirm.showModal !== "function") {
    return Promise.resolve(window.confirm(`Delete session?\n\n${text}`));
  }
  els.confirmText.textContent = text;
  els.confirm.returnValue = "cancel"; // what Escape and the backdrop leave behind
  els.confirm.showModal();
  return new Promise((resolve) => {
    els.confirm.addEventListener("close", () => resolve(els.confirm.returnValue === "confirm"), {
      once: true,
    });
  });
}

async function deleteSession(session) {
  if (inCall() || !(await confirmDelete(session))) return;
  try {
    const wasCurrent = state.session?.id === session.id;
    // Never leave the list empty: the replacement keeps the deleted session's settings.
    let next = state.sessions.find((s) => s.id !== session.id)?.id;
    if (wasCurrent && next === undefined) {
      next = (await api("POST", "/api/sessions", { copy_from: session.id })).id;
    }
    await api("DELETE", `/api/sessions/${session.id}`);
    setNotice("sessions", "error", null);
    if (wasCurrent) {
      state.session = null;
      await selectSession(next);
    }
  } catch (err) {
    setNotice("sessions", "error", `Could not delete the session: ${err.message}`);
  }
  await loadSessions().catch(() => {});
}

// -- settings --

function fillSelect(select, values, current, label) {
  select.replaceChildren();
  const all = values.includes(current) ? values : [current, ...values];
  for (const value of all) {
    const option = el("option", "", label(value));
    option.value = value;
    if (!values.includes(value)) option.textContent += " (not available)";
    select.append(option);
  }
  select.value = current;
}

function renderSettings() {
  const s = state.session;
  fillSelect(els.model, state.models, s.model, shortModel);
  fillSelect(els.voice, state.voices, s.voice, shortVoice);
  if (document.activeElement !== els.prompt) els.prompt.value = s.system_prompt;
  els.savePrompt.disabled = els.prompt.value === s.system_prompt;
  els.settingsSummary.textContent = `${shortModel(s.model)} · ${shortVoice(s.voice)}`;
  els.settingsNote.textContent = inCall() ? "Changes apply from the next turn." : "";
  renderSettingsNotice();
}

// The session's model or voice is no longer offered (only known while the lists load).
function renderSettingsNotice() {
  const s = state.session;
  const missing = [];
  if (state.models.length && !state.models.includes(s.model)) {
    missing.push(`model “${shortModel(s.model)}” is not installed in Ollama`);
  }
  if (state.voices.length && !state.voices.includes(s.voice)) {
    missing.push(`voice “${shortVoice(s.voice)}” is not offered by Chatterbox`);
  }
  const text = missing.length
    ? `This session's ${missing.join(" and its ")}. Pick another in Settings.`
    : null;
  setNotice("settings", "warn", text);
}

function renderSession() {
  const s = state.session;
  els.title.value = s.title ?? "";
  document.title = s.title ? `${s.title} · voicedm` : "voicedm";
  renderSettings();
  renderTranscript();
}

async function patchSession(changes) {
  const id = state.session.id;
  try {
    const updated = await api("PATCH", `/api/sessions/${id}`, changes);
    if (state.session.id !== id) return;
    // Keep the live transcript; take the settings and title from the server.
    const { messages, ...fields } = updated;
    Object.assign(state.session, fields);
    if (!inCall()) state.session.messages = messages;
    renderSettings();
    await loadSessions();
    if (inCall() && ("model" in changes || "voice" in changes || "system_prompt" in changes)) {
      els.settingsNote.textContent = "Saved; applies from the next turn.";
    }
  } catch (err) {
    els.settingsNote.textContent = `Not saved: ${err.message}`;
    renderSettings();
  }
}

// -- transcript --

function messageNode(role, text, { interrupted = false, model = null } = {}) {
  const item = el("li", `message ${role}`);
  const body = el("span", "body", text);
  item.append(body);
  if (interrupted) item.classList.add("interrupted");
  const meta = [];
  if (model) meta.push(shortModel(model));
  if (interrupted) meta.push("interrupted");
  if (meta.length) item.append(el("span", "meta", meta.join(" · ")));
  return item;
}

function renderTranscript() {
  els.transcript.replaceChildren();
  const messages = state.session?.messages ?? [];
  if (!messages.length && !state.live) {
    const hint = el("li", "empty", "No messages yet. Press Call and start talking.");
    els.transcript.append(hint);
    return;
  }
  if (state.session?.summary) els.transcript.append(summaryNode(state.session.summary));
  for (const m of messages) {
    els.transcript.append(
      messageNode(m.role, m.content, { interrupted: m.interrupted, model: m.model }),
    );
  }
  scrollTranscript();
}

// What the bot remembers of the messages folded out of its context; the messages
// themselves are all still listed.
function summaryNode(text) {
  const item = el("li", "memory");
  const details = el("details");
  details.append(el("summary", null, "Earlier conversation summary"), el("p", null, text));
  item.append(details);
  return item;
}

function scrollTranscript() {
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

// Live transcript: user speech as Whisper transcribes it, and the bot's reply as the
// LLM generates it. Text not yet spoken is shown muted; it turns solid as the
// server reports each sentence spoken. An interrupted reply keeps the spoken part.

function newLive() {
  return { user: null, bot: null };
}

function removeEmptyHint() {
  els.transcript.querySelector(".empty")?.remove();
}

function addUserText(text) {
  const live = state.live;
  if (!live || !text.trim()) return;
  if (live.bot) finishBotReply(false);
  if (!live.user) {
    removeEmptyHint();
    live.user = messageNode("user", "");
    els.transcript.append(live.user);
  }
  const body = live.user.querySelector(".body");
  body.textContent = `${body.textContent} ${text.trim()}`.trim();
  scrollTranscript();
}

function botReply() {
  const live = state.live;
  if (!live.bot) {
    removeEmptyHint();
    const node = messageNode("assistant", "");
    const pending = el("span", "pending");
    node.querySelector(".body").after(pending);
    live.bot = { node, generated: "", spoken: "", llmActive: false };
    live.user = null;
    els.transcript.append(node);
  }
  return live.bot;
}

// The part of `generated` after the text matching `spoken`, ignoring whitespace.
function unspokenTail(generated, spoken) {
  const said = spoken.replace(/\s+/g, "");
  let i = 0;
  let j = 0;
  while (i < generated.length && j < said.length) {
    if (/\s/.test(generated[i])) {
      i++;
    } else if (generated[i] === said[j]) {
      i++;
      j++;
    } else {
      break;
    }
  }
  return generated.slice(i);
}

function renderBotReply(reply) {
  reply.node.querySelector(".body").textContent = reply.spoken;
  const tail = unspokenTail(reply.generated, reply.spoken);
  reply.node.querySelector(".pending").textContent = reply.spoken ? tail : tail.trimStart();
  scrollTranscript();
}

function replyInProgress() {
  const reply = state.live?.bot;
  if (!reply) return false;
  return reply.llmActive || unspokenTail(reply.generated, reply.spoken).trim() !== "";
}

function finishBotReply(interrupted) {
  const reply = state.live?.bot;
  if (!reply) return;
  state.live.bot = null;
  reply.node.querySelector(".pending").remove();
  if (!reply.spoken.trim()) {
    reply.node.remove();
    return;
  }
  if (interrupted) {
    reply.node.classList.add("interrupted");
    reply.node.append(el("span", "meta", "interrupted"));
  }
}

// -- call --

function waitForIceGathering(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    peer.addEventListener("icegatheringstatechange", () => {
      if (peer.iceGatheringState === "complete") resolve();
    });
  });
}

function scheduleListening() {
  clearTimeout(state.listenTimer);
  state.listenTimer = setTimeout(() => {
    if (!replyInProgress()) setCallState("listening");
  }, LISTEN_DELAY_MS);
}

function onServerMessage(call, event) {
  // Messages of a call that has been replaced or ended are stale.
  if (state.call !== call || !state.live) return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (msg.label !== "rtvi-ai") return;
  const data = msg.data ?? {};
  switch (msg.type) {
    case "server-message":
      onAppMessage(call, data);
      break;
    case "user-started-speaking":
      if (replyInProgress()) finishBotReply(true);
      setCallState("listening");
      break;
    case "user-stopped-speaking":
      setCallState("thinking");
      break;
    case "user-transcription":
      if (data.final) addUserText(data.text ?? "");
      break;
    case "bot-llm-started":
      botReply().llmActive = true;
      break;
    case "bot-llm-text": {
      const reply = botReply();
      reply.generated += data.text ?? "";
      renderBotReply(reply);
      break;
    }
    case "bot-llm-stopped":
      if (state.live.bot) state.live.bot.llmActive = false;
      if (state.callState === "speaking" && !replyInProgress()) scheduleListening();
      break;
    case "bot-tts-text": {
      const reply = botReply();
      reply.spoken = `${reply.spoken} ${(data.text ?? "").trim()}`.trim();
      renderBotReply(reply);
      break;
    }
    case "bot-started-speaking":
      setCallState("speaking");
      break;
    case "bot-stopped-speaking":
      // Between sentences, keep "Speaking" while the reply has more to say.
      if (replyInProgress()) break;
      finishBotReply(false);
      scheduleListening();
      break;
    case "error":
      // Pipecat's raw error; the server follows it with an "error" server message
      // that says what went wrong in plain words.
      console.warn("voicedm: pipeline error", data.error);
      break;
  }
}

// The server's own messages: warm-up progress, errors, and why it ended a call.
function onAppMessage(call, data) {
  switch (data.type) {
    case "warmup":
      if (data.state === "warming") {
        setWarming(data.pending ?? ["ollama", "chatterbox"]);
        if (state.callState === "connecting") setCallState("warming");
      } else {
        setWarming([]);
        const failed = Object.entries(data.errors ?? {});
        const down = ["ollama", "chatterbox"].filter((name) => data[name] === false);
        const text = failed.length
          ? failed.map(([, message]) => message).join(" ")
          : down.length
            ? `Could not warm up ${down.map((d) => UPSTREAM_NAMES[d]).join(" and ")}.`
            : null;
        setNotice("warmup", "error", text);
        if (["warming", "connecting"].includes(state.callState)) setCallState("listening");
      }
      break;
    case "error":
      setNotice(`call-${data.service}`, "error", data.message);
      // A reply that failed leaves nothing to wait for.
      if (data.service === "llm" && state.callState === "thinking") setCallState("listening");
      break;
    case "call-ended":
      call.ended = true;
      if (data.retry) {
        dropped(call, data.message ?? "The server ended the call");
      } else {
        endCall("Call ended", "error");
        // The reason replaces what the call reported on its way there.
        for (const key of state.notices.keys()) {
          if (key.startsWith("call-") || key === "warmup") state.notices.delete(key);
        }
        setNotice("call", "error", data.message ?? "The server ended the call.");
      }
      break;
  }
}

function micError(err) {
  switch (err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access was denied. Allow it for this site in the browser's settings, then press Call again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found.";
    case "NotReadableError":
      return "The microphone could not be opened; another app may be using it.";
    default:
      return `The microphone could not be opened: ${err.message}`;
  }
}

// Browsers only give pages on a secure origin (HTTPS or localhost) the microphone.
function micUnavailable() {
  if (window.isSecureContext && navigator.mediaDevices?.getUserMedia) return null;
  return (
    "The microphone is only available on a secure page. Open voicedm over HTTPS " +
    "(for example through a reverse proxy) or on localhost."
  );
}

// Pressing Call: the microphone, then a connection.
async function call() {
  if (!state.session || inCall()) return;
  const blocked = micUnavailable();
  if (blocked) {
    setNotice("mic", "error", blocked);
    return;
  }
  setNotice("mic");
  for (const key of state.notices.keys()) {
    if (key.startsWith("call") || key === "warmup") setNotice(key);
  }
  els.call.disabled = true;
  setCallState("connecting");
  state.live = newLive();
  state.call = { sessionId: state.session.id, pcId: null };
  renderCallControls();
  const call = state.call;
  let mic;
  try {
    mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    if (state.call === call) endCall("Call failed", "error");
    setNotice("mic", "error", micError(err));
    return;
  } finally {
    els.call.disabled = false;
  }
  if (state.call !== call) {
    // Hung up while the browser was asking for the microphone.
    for (const track of mic.getTracks()) track.stop();
    return;
  }
  state.mic = mic;
  applyMute();
  try {
    await connect(call);
  } catch (err) {
    if (state.call !== call) return;
    if (err.status === 409) {
      endCall("Busy", "error");
      setNotice("call", "error", err.message);
      refreshHealth();
    } else {
      endCall("Call failed", "error");
      setNotice("call", "error", `The call could not be started: ${err.message}`);
    }
  }
}

// Open a peer connection for `call` and exchange the offer; on a reconnect the
// server ends the dropped call (`replaces`) before answering.
async function connect(call, replaces = null) {
  const pc = new RTCPeerConnection();
  call.pc = pc;
  call.ended = false;
  state.pc = pc;
  pc.addTransceiver(state.mic.getAudioTracks()[0], { direction: "sendrecv" });
  pc.addEventListener("track", (event) => {
    els.remote.srcObject = event.streams[0] ?? new MediaStream([event.track]);
    els.remote.play().catch(() => {});
  });
  let graceTimer = null;
  pc.addEventListener("connectionstatechange", () => {
    if (state.pc !== pc) return;
    clearTimeout(graceTimer);
    if (pc.connectionState === "disconnected") {
      // Often recovers by itself; give it a moment.
      graceTimer = setTimeout(() => {
        if (state.pc === pc && pc.connectionState !== "connected") dropped(call);
      }, DISCONNECT_GRACE_MS);
    } else if (["failed", "closed"].includes(pc.connectionState)) {
      dropped(call);
    }
  });

  const channel = pc.createDataChannel("messages", { ordered: true });
  channel.addEventListener("open", () => {
    channel.send(
      JSON.stringify({
        label: "rtvi-ai",
        type: "client-ready",
        id: crypto.randomUUID(),
        data: { version: "2.3.0", about: { library: "voicedm-page" } },
      }),
    );
  });
  channel.addEventListener("message", (event) => onServerMessage(call, event));
  // The server closes the channel when it ends the call; unless it said why
  // (call-ended, handled first: the channel is ordered), the call dropped.
  channel.addEventListener("close", () => {
    if (state.pc === pc && !call.ended) dropped(call);
  });

  try {
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIceGathering(pc);
    const answer = await api("POST", "/api/offer", {
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      session_id: call.sessionId,
      replaces,
    });
    if (state.pc !== pc) return; // hung up meanwhile
    call.pcId = answer.pc_id;
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
  } catch (err) {
    if (state.pc === pc) state.pc = null;
    pc.close();
    throw err;
  }
}

// The connection of `call` is gone without a hang-up: tell the user and reconnect
// onto the same session, with a growing pause between attempts.
function dropped(call, why = "Connection lost") {
  if (state.call !== call || state.reconnect) return;
  const peer = call.pc;
  if (state.pc === peer) state.pc = null;
  peer?.close();
  finishBotReply(true);
  setWarming([]);
  els.remote.srcObject = null;
  why = why.replace(/[.\s]+$/, "");
  state.reconnect = { attempt: 0, timer: null, why, replaces: call.pcId };
  scheduleReconnect(call);
}

function scheduleReconnect(call) {
  const r = state.reconnect;
  if (r.attempt >= RECONNECT_DELAYS_MS.length) {
    endCall("Call dropped", "error");
    setNotice("call", "error", `${r.why}, and reconnecting failed. Press Call to try again.`);
    return;
  }
  const delay = RECONNECT_DELAYS_MS[r.attempt];
  r.attempt += 1;
  setCallState("reconnecting", `Reconnecting (${r.attempt}/${RECONNECT_DELAYS_MS.length})…`);
  setNotice("call", "warn", `${r.why}. Reconnecting to this session…`);
  r.timer = setTimeout(() => reconnect(call), delay);
}

async function reconnect(call) {
  const r = state.reconnect;
  if (state.call !== call || !r) return;
  try {
    await connect(call, r.replaces);
  } catch (err) {
    if (state.call !== call || state.reconnect !== r) return; // hung up meanwhile
    // Another call took over (409), or the session is gone: stop trying.
    if (err.status && err.status < 500 && err.status !== 408) {
      endCall("Call dropped", "error");
      setNotice("call", "error", `Could not reconnect: ${err.message}`);
      if (err.status === 409) refreshHealth();
      return;
    }
    scheduleReconnect(call);
    return;
  }
  if (state.call !== call) return;
  state.reconnect = null;
  setNotice("call");
  setCallState("connecting");
  // The dropped call's turns are stored by now; show them from the server and
  // start a fresh live transcript, so nothing is listed twice.
  try {
    const session = await api("GET", `/api/sessions/${call.sessionId}`);
    if (state.call === call && state.session?.id === call.sessionId) {
      state.session = session;
      finishBotReply(true);
      state.live = newLive();
      renderTranscript();
    }
  } catch {
    state.live = newLive();
  }
}

function renderCallControls() {
  els.call.textContent = inCall() ? "Hang up" : "Call";
  els.call.classList.toggle("active", inCall());
  els.mute.disabled = !inCall();
  renderAvailability();
  renderSessions();
  if (state.session) renderSettings();
}

// Pressing Hang up: never followed by a reconnect.
function hangUp() {
  endCall("Call ended");
}

function endCall(message = "Call ended", stateName = "idle") {
  const call = state.call;
  state.call = null;
  if (state.reconnect) {
    clearTimeout(state.reconnect.timer);
    state.reconnect = null;
    setNotice("call");
  }
  // The server lets go of the call in a moment; until the next check, the last one
  // would still show it as running elsewhere.
  if (call && state.health?.call) state.health = { ...state.health, call: null };
  if (state.pc) {
    const peer = state.pc;
    state.pc = null;
    peer.close();
  }
  if (state.mic) {
    for (const track of state.mic.getTracks()) track.stop();
    state.mic = null;
  }
  finishBotReply(true);
  state.live = null;
  setWarming([]);
  els.remote.srcObject = null;
  setCallState(stateName, message);
  renderCallControls();
  if (call && state.session) {
    // Replace the live transcript with what the server stored.
    const id = state.session.id;
    setTimeout(async () => {
      if (inCall() || state.session?.id !== id) return;
      await selectSession(id).catch(() => {});
      await loadSessions().catch(() => {});
    }, RELOAD_DELAY_MS);
  }
  // Once the server has let go of the call.
  setTimeout(refreshHealth, RELOAD_DELAY_MS);
}

function applyMute() {
  for (const track of state.mic?.getAudioTracks() ?? []) track.enabled = !state.muted;
  els.mute.setAttribute("aria-pressed", String(state.muted));
  els.mute.textContent = state.muted ? "Unmute" : "Mute";
}

// -- upstream health --

// Check the upstreams and the server's call; reschedules itself while idle.
async function refreshHealth() {
  clearTimeout(state.healthTimer);
  const before = state.health;
  try {
    state.health = await api("GET", "/api/health");
  } catch {
    state.health = { status: "down" };
  }
  const h = state.health;
  // An upstream that came back: reload what it lists.
  if (before && h.upstreams) {
    if (h.upstreams.ollama && !before.upstreams?.ollama) loadModels();
    if (h.upstreams.chatterbox && !before.upstreams?.chatterbox) loadVoices();
  }
  renderAvailability();
  if (h.call && !inCall()) loadSessions().catch(() => {});
  const down = h.status !== "ok" || !h.upstreams?.ollama;
  state.healthTimer = setTimeout(refreshHealth, down ? HEALTH_POLL_DOWN_MS : HEALTH_POLL_MS);
}

function renderAvailability() {
  const h = state.health;
  const mic = micUnavailable();
  let reason = null;
  if (!h) {
    reason = null;
  } else if (h.status !== "ok") {
    reason = "The voicedm server is not reachable.";
  } else if (!h.upstreams.ollama) {
    reason = "The language model server (Ollama) is not reachable, so a call cannot start. Checking again every few seconds.";
  }
  if (!inCall()) {
    els.call.disabled = Boolean(reason || mic);
    els.call.title = reason ?? mic ?? "";
  }
  setNotice("ollama", "error", inCall() ? null : reason);
  setNotice(
    "chatterbox",
    "warn",
    h?.status === "ok" && !h.upstreams.chatterbox
      ? "The voice server (Chatterbox) is not answering. It may still be starting; calls can fail until it does."
      : null,
  );
  const elsewhere = h?.call && !inCall() ? h.call.session_id : null;
  setNotice(
    "busy",
    "info",
    elsewhere !== null
      ? `A call is running on “${sessionTitle(elsewhere)}” in another tab or on another device.`
      : null,
  );
  if (mic) setNotice("mic", "error", mic);
}

async function loadModels() {
  try {
    state.models = await api("GET", "/api/models");
    setNotice(
      "models",
      "error",
      state.models.length ? null : "Ollama has no models installed. Pull one with “ollama pull”.",
    );
  } catch (err) {
    state.models = [];
    setNotice("models", "warn", `Cannot list the models: ${err.message}`);
  }
  if (state.session) renderSettings();
}

async function loadVoices() {
  try {
    state.voices = await api("GET", "/api/voices");
    setNotice("voices", "error", state.voices.length ? null : "Chatterbox lists no voices.");
  } catch (err) {
    state.voices = [];
    setNotice("voices", "warn", `Cannot list the voices: ${err.message}`);
  }
  if (state.session) renderSettings();
}

// -- drawer (phone width) --

function openDrawer() {
  els.sidebar.classList.add("open");
  els.scrim.hidden = false;
  els.menu.setAttribute("aria-expanded", "true");
}

function closeDrawer() {
  els.sidebar.classList.remove("open");
  els.scrim.hidden = true;
  els.menu.setAttribute("aria-expanded", "false");
}

// -- wiring --

els.call.addEventListener("click", () => (inCall() ? hangUp() : call()));
els.mute.addEventListener("click", () => {
  state.muted = !state.muted;
  applyMute();
});
els.newSession.addEventListener("click", newSession);
els.menu.addEventListener("click", openDrawer);
els.scrim.addEventListener("click", closeDrawer);
// A click on the backdrop (the dialog element itself, outside its form) cancels.
els.confirm.addEventListener("click", (event) => {
  if (event.target === els.confirm) els.confirm.close("cancel");
});

els.model.addEventListener("change", () => patchSession({ model: els.model.value }));
els.voice.addEventListener("change", () => patchSession({ voice: els.voice.value }));
els.prompt.addEventListener("input", () => {
  els.savePrompt.disabled = els.prompt.value === state.session?.system_prompt;
});
els.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) els.savePrompt.click();
});
els.savePrompt.addEventListener("click", () =>
  patchSession({ system_prompt: els.prompt.value }),
);

els.title.addEventListener("keydown", (event) => {
  if (event.key === "Enter") els.title.blur();
  if (event.key === "Escape") {
    els.title.value = state.session?.title ?? "";
    els.title.blur();
  }
});
els.title.addEventListener("change", () => {
  if (state.session && els.title.value.trim() !== (state.session.title ?? "")) {
    patchSession({ title: els.title.value.trim() });
  }
});

window.addEventListener("hashchange", () => {
  const id = Number(location.hash.slice(1));
  if (id && id !== state.session?.id && !inCall()) {
    selectSession(id).catch((err) => setCallState("error", err.message));
  }
});

async function init() {
  renderAvailability();
  await Promise.all([loadModels(), loadVoices(), refreshHealth()]);
  try {
    await loadSessions();
    if (!state.sessions.length) {
      await api("POST", "/api/sessions");
      await loadSessions();
    }
    const wanted = Number(location.hash.slice(1));
    const id = state.sessions.some((s) => s.id === wanted) ? wanted : state.sessions[0].id;
    await selectSession(id);
    renderAvailability();
  } catch (err) {
    setCallState("error", "Cannot load sessions");
    setNotice("sessions", "error", `Cannot load sessions: ${err.message}`);
  }
}

// Check the upstreams again when the page comes back into view.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && !inCall()) refreshHealth();
});
window.addEventListener("focus", () => {
  if (!inCall()) refreshHealth();
});

init();
