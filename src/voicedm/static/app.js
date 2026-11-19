"use strict";

// Bare call page: one RTCPeerConnection to the server, mic out, bot audio in.
// Server events arrive as RTVI messages on the data channel.

const button = document.getElementById("call");
const statusLine = document.getElementById("status");
const remoteAudio = document.getElementById("remote");

let pc = null;
let mic = null;

function setStatus(text) {
  statusLine.textContent = text;
}

function waitForIceGathering(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    peer.addEventListener("icegatheringstatechange", () => {
      if (peer.iceGatheringState === "complete") resolve();
    });
  });
}

function onServerMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (msg.label !== "rtvi-ai") return;
  switch (msg.type) {
    case "server-message":
      if (msg.data?.type === "warmup") {
        if (msg.data.state === "warming") {
          setStatus("Warming up…");
        } else {
          const down = ["ollama", "chatterbox"].filter((name) => !msg.data[name]);
          setStatus(down.length ? `Warm-up failed: ${down.join(", ")}` : "Listening");
        }
      }
      break;
    case "user-started-speaking":
      setStatus("Listening");
      break;
    case "user-stopped-speaking":
      setStatus("Thinking");
      break;
    case "bot-started-speaking":
      setStatus("Speaking");
      break;
    case "bot-stopped-speaking":
      setStatus("Listening");
      break;
    case "error":
      setStatus(`Error: ${msg.data?.error ?? "unknown"}`);
      break;
  }
}

async function call() {
  button.disabled = true;
  setStatus("Connecting…");
  try {
    mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    pc = new RTCPeerConnection();
    pc.addTransceiver(mic.getAudioTracks()[0], { direction: "sendrecv" });
    pc.addEventListener("track", (event) => {
      remoteAudio.srcObject = event.streams[0] ?? new MediaStream([event.track]);
      remoteAudio.play().catch(() => {});
    });
    pc.addEventListener("connectionstatechange", () => {
      if (pc && ["failed", "closed"].includes(pc.connectionState)) {
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
    const response = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `HTTP ${response.status}`);
    }
    const answer = await response.json();
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });

    button.textContent = "Hang up";
    button.classList.add("active");
    setStatus("Connected");
  } catch (err) {
    hangUp(`Call failed: ${err.message}`);
  } finally {
    button.disabled = false;
  }
}

function hangUp(message = "Idle") {
  if (pc) {
    const peer = pc;
    pc = null;
    peer.close();
  }
  if (mic) {
    for (const track of mic.getTracks()) track.stop();
    mic = null;
  }
  remoteAudio.srcObject = null;
  button.textContent = "Call";
  button.classList.remove("active");
  setStatus(message);
}

button.addEventListener("click", () => (pc ? hangUp() : call()));
