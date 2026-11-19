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
};

// Pause after "bot stopped speaking" before showing "Listening": the server reports
// a stop between sentences whenever speech synthesis is slower than playback.
const LISTEN_DELAY_MS = 700;
// Time for the server to save the last turn after a hang-up before reloading.
const RELOAD_DELAY_MS = 1200;

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
    const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    throw new Error(detail ?? `HTTP ${response.status}`);
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

function inCall() {
  return state.pc !== null;
}

// -- status --

const STATUS_TEXT = {
  idle: "Idle",
  connecting: "Connecting…",
  warming: "Warming up…",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
};

function setCallState(name, text) {
  clearTimeout(state.listenTimer);
  state.listenTimer = null;
  state.callState = name;
  els.dot.dataset.state = name;
  els.status.textContent = text ?? STATUS_TEXT[name] ?? name;
}

// -- sessions --

function renderSessions() {
  els.sessions.replaceChildren();
  for (const session of state.sessions) {
    const button = el("button");
    button.type = "button";
    button.append(
      el("span", "session-title", session.title || "Untitled session"),
      el(
        "span",
        "session-meta",
        `${session.message_count} messages · ${relativeTime(session.updated_at)}`,
      ),
    );
    const current = state.session?.id === session.id;
    button.setAttribute("aria-current", String(current));
    button.disabled = inCall() && !current;
    button.addEventListener("click", () => {
      closeDrawer();
      if (!current) selectSession(session.id);
    });
    const item = el("li");
    item.append(button);
    els.sessions.append(item);
  }
  els.newSession.disabled = inCall();
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

function onServerMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (msg.label !== "rtvi-ai" || !state.live) return;
  const data = msg.data ?? {};
  switch (msg.type) {
    case "server-message":
      if (data.type === "warmup") {
        if (data.state === "warming") {
          setCallState("warming");
        } else {
          const down = ["ollama", "chatterbox"].filter((name) => !data[name]);
          if (down.length) setCallState("error", `Warm-up failed: ${down.join(", ")}`);
          else if (state.callState === "warming") setCallState("listening");
        }
      }
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
      setCallState("error", `Error: ${data.error ?? "unknown"}`);
      break;
  }
}

async function call() {
  if (!state.session) return;
  els.call.disabled = true;
  setCallState("connecting");
  state.live = newLive();
  try {
    state.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    applyMute();
    const pc = new RTCPeerConnection();
    state.pc = pc;
    pc.addTransceiver(state.mic.getAudioTracks()[0], { direction: "sendrecv" });
    pc.addEventListener("track", (event) => {
      els.remote.srcObject = event.streams[0] ?? new MediaStream([event.track]);
      els.remote.play().catch(() => {});
    });
    pc.addEventListener("connectionstatechange", () => {
      if (state.pc === pc && ["failed", "closed"].includes(pc.connectionState)) {
        hangUp(`Call ${pc.connectionState}`);
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
    channel.addEventListener("message", onServerMessage);

    await pc.setLocalDescription(await pc.createOffer());
    await waitForIceGathering(pc);
    const answer = await api("POST", "/api/offer", {
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      session_id: state.session.id,
    });
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });

    els.call.textContent = "Hang up";
    els.call.classList.add("active");
    els.mute.disabled = false;
    renderSessions();
    renderSettings();
  } catch (err) {
    hangUp(`Call failed: ${err.message}`, "error");
  } finally {
    els.call.disabled = false;
  }
}

function hangUp(message = "Call ended", stateName = "idle") {
  const wasInCall = state.pc !== null;
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
  els.remote.srcObject = null;
  els.call.textContent = "Call";
  els.call.classList.remove("active");
  els.mute.disabled = true;
  setCallState(stateName, message);
  renderSessions();
  if (state.session) renderSettings();
  if (wasInCall && state.session) {
    // Replace the live transcript with what the server stored.
    const id = state.session.id;
    setTimeout(async () => {
      if (inCall() || state.session?.id !== id) return;
      await selectSession(id).catch(() => {});
      await loadSessions().catch(() => {});
    }, RELOAD_DELAY_MS);
  }
}

function applyMute() {
  for (const track of state.mic?.getAudioTracks() ?? []) track.enabled = !state.muted;
  els.mute.setAttribute("aria-pressed", String(state.muted));
  els.mute.textContent = state.muted ? "Unmute" : "Mute";
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
  const [models, voices] = await Promise.allSettled([
    api("GET", "/api/models"),
    api("GET", "/api/voices"),
  ]);
  state.models = models.status === "fulfilled" ? models.value : [];
  state.voices = voices.status === "fulfilled" ? voices.value : [];
  const down = [];
  if (models.status === "rejected") down.push("Ollama");
  if (voices.status === "rejected") down.push("Chatterbox");

  try {
    await loadSessions();
    if (!state.sessions.length) {
      await api("POST", "/api/sessions");
      await loadSessions();
    }
    const wanted = Number(location.hash.slice(1));
    const id = state.sessions.some((s) => s.id === wanted) ? wanted : state.sessions[0].id;
    await selectSession(id);
    if (down.length) setCallState("error", `Cannot reach ${down.join(" and ")}`);
  } catch (err) {
    setCallState("error", `Cannot load sessions: ${err.message}`);
  }
}

init();
